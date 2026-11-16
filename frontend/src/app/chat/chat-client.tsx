"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  api,
  type ChatLogTurn,
  type ChatRequest,
  type ChatResponse,
  type ChatSession,
  type CommandKind,
  type CommandStatus,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { EmotionMeter } from "@/components/emotion-meter";

type ChatTurn =
  | { role: "user" | "assistant"; content: string; result?: ChatResponse; turnId?: number; failed?: string }
  | { role: "system"; ticketId: string; kind: CommandKind; status: string };

// #370 D1 2-4: poll from 1 s, doubling, for at most 60 s in total.
export const FIRST_POLL_MS = 1_000;
export const POLL_LIMIT_MS = 60_000;

const KIND_LABEL: Record<CommandKind, string> = {
  REVIEW_EXPERIENCE: "Review request",
  COMMIT_REVIEW_EXPERIENCE: "Review commitment",
};

// Settled statuses end polling; "pending" keeps it going.
const FINAL = new Set(["adopted", "accepted", "hold", "unavailable", "dropped", "unknown_after_restart", "not_found"]);

const STATUS_LABEL: Record<string, string> = {
  not_found: "not found (the result is no longer kept)",
  pending: "in progress",
  adopted: "adopted as a Goal",
  accepted: "accepted as a Commitment",
  hold: "on hold",
  unavailable: "unavailable",
  dropped: "dropped",
  unknown_after_restart: "unknown (the server restarted)",
};

// Per-viewer convenience: which session to reopen. Not authoritative state.
export const SESSION_STORAGE_KEY = "suzka.chat.session";

function storedSession(): string | null {
  try {
    return window.localStorage.getItem(SESSION_STORAGE_KEY);
  } catch {
    return null;
  }
}

function storeSession(sessionId: string | null) {
  try {
    if (sessionId) window.localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
    else window.localStorage.removeItem(SESSION_STORAGE_KEY);
  } catch {
    // Storage may be unavailable (private window); the screen still works.
  }
}

function turnsFromLog(turns: ChatLogTurn[]): ChatTurn[] {
  // Restored turns carry no response metadata, so they offer no review actions.
  return turns.map((turn) => ({ role: turn.role, content: turn.content }));
}

export function reviewCommand(kind: CommandKind, experienceId: string): string {
  // The screen builds the exact command text the owners check (DRV-3).
  return kind === "REVIEW_EXPERIENCE"
    ? `REVIEW_EXPERIENCE:${experienceId}`
    : `COMMIT_REVIEW_EXPERIENCE:1:${experienceId}`;
}

export function ChatClient() {
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState<ChatTurn[]>([]);
  // The composer text that was just sent, so a failed send can put it back.
  const sentDraft = useRef<string | null>(null);
  const nextTurnId = useRef(0);
  const [sessionId, setSessionId] = useState<string | null>(null);
  // The mutation reads the session through a ref so review buttons on an
  // already-rendered turn still post to the current session.
  const sessionRef = useRef<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const sessions = useQuery({ queryKey: ["chat-sessions"], queryFn: api.chatSessions });

  const selectSession = (id: string | null) => {
    sessionRef.current = id;
    setSessionId(id);
    storeSession(id);
  };

  // Bumped by every load and every send: a load whose token is no longer
  // current (a later click, or a message sent meanwhile) is dropped.
  const loadToken = useRef(0);

  const openSession = async (id: string) => {
    const token = ++loadToken.current;
    setLoadError(null);
    try {
      const detail = await api.chatSession(id);
      if (token !== loadToken.current) return;
      selectSession(id);
      setHistory(turnsFromLog(detail.turns));
    } catch (error) {
      if (token !== loadToken.current) return;
      if (error instanceof ApiError && error.status === 404) {
        if (storedSession() === id) storeSession(null);
        setLoadError("That chat no longer exists.");
      } else {
        setLoadError(error instanceof Error ? error.message : "Could not load the chat.");
      }
    }
  };

  useEffect(() => {
    const id = storedSession();
    if (id) void openSession(id);
    // Only on first render: reopen the last session this viewer used.
  }, []);

  const startNewChat = () => {
    loadToken.current += 1;
    selectSession(null);
    setHistory([]);
    setLoadError(null);
  };

  const deleteSession = async (session: ChatSession) => {
    if (!window.confirm(`Delete the chat "${session.title || "Untitled"}"? Only its transcript is removed; SUZKA's memory is not.`)) return;
    try {
      await api.deleteChatSession(session.session_id);
    } catch (error) {
      if (!(error instanceof ApiError && error.status === 404)) {
        setLoadError(error instanceof Error ? error.message : "Could not delete the chat.");
        return;
      }
    }
    if (sessionRef.current === session.session_id) startNewChat();
    void queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
  };

  const mutation = useMutation({
    mutationFn: async (body: ChatRequest) => {
      let id = sessionRef.current;
      let unsaved = false;
      if (!id) {
        // The first message of a new chat opens its session. The transcript
        // is display-only, so failing to open one never blocks the turn.
        try {
          const created = await api.createChatSession();
          if (typeof created?.session_id === "string") {
            id = created.session_id;
            selectSession(id);
          }
        } catch {
          unsaved = true;
        }
      }
      let result: ChatResponse;
      try {
        result = await api.chat(id ? { ...body, chat_session_id: id } : body);
      } catch (error) {
        // The session was deleted elsewhere: leave it, so the next send opens a new one.
        if (id && error instanceof ApiError && error.status === 404) {
          selectSession(null);
          throw new Error("This chat was deleted elsewhere. Send again to start a new chat.");
        }
        throw error;
      }
      return unsaved ? { ...result, transcript_saved: false } : result;
    },
    // The user's turn shows as soon as it is sent; the reply follows it.
    onMutate: (variables) => {
      loadToken.current += 1;
      const turnId = ++nextTurnId.current;
      setHistory((current) => [...current, { role: "user", content: variables.message, turnId }]);
      return { turnId };
    },
    onError: (error, variables, context) => {
      // Mark the turn that did not get through, and give the draft back.
      if (context) {
        setHistory((current) =>
          current.map((turn) =>
            "turnId" in turn && turn.turnId === context.turnId ? { ...turn, failed: error.message } : turn,
          ),
        );
      }
      if (sentDraft.current === variables.message) {
        setMessage((draft) => draft || variables.message);
      }
    },
    onSettled: () => {
      sentDraft.current = null;
      void queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
    },
    onSuccess: (result) => {
      const ticket = result.command_ticket;
      setHistory((current) => [
        ...current,
        { role: "assistant", content: result.response, result },
        ...(ticket
          ? [{ role: "system" as const, ticketId: ticket.ticket_id, kind: ticket.kind, status: ticket.status }]
          : []),
      ]);
    },
  });

  const latest = [...history]
    .reverse()
    .find((turn): turn is Extract<ChatTurn, { result?: ChatResponse }> => "result" in turn && !!turn.result)?.result;
  // The server incarnation of the latest response: a different value means
  // the server restarted, so earlier turns can no longer be reviewed.
  const incarnation = latest?.incarnation ?? null;

  const send = (text: string) => mutation.mutate({ message: text, attachments: [], debug: false });
  const submitDraft = () => {
    if (!message.trim() || mutation.isPending) return;
    sentDraft.current = message;
    send(message);
    setMessage("");
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1 className="page-title">Chat</h1>
          <p className="page-subtitle">Normal chat shows only user-facing answers and emotion state.</p>
        </div>
        {latest ? <Badge>{latest.model.model_id}</Badge> : null}
      </header>

      <Card className="chat-sessions" aria-label="Chats">
        <div className="chat-sessions-header">
          <CardTitle>Chats</CardTitle>
          <Button type="button" disabled={mutation.isPending} onClick={startNewChat}>New chat</Button>
        </div>
        <ul className="chat-session-list">
          {(sessions.data?.sessions ?? []).map((session) => (
            <li key={session.session_id} className={session.session_id === sessionId ? "active" : undefined}>
              <button
                type="button"
                className="chat-session-open"
                disabled={mutation.isPending}
                aria-current={session.session_id === sessionId ? "true" : undefined}
                onClick={() => void openSession(session.session_id)}
              >
                {session.title || "Untitled"}
              </button>
              <button
                type="button"
                className="chat-session-delete"
                disabled={mutation.isPending}
                aria-label={`Delete ${session.title || "Untitled"}`}
                onClick={() => void deleteSession(session)}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
        {loadError ? <p className="error">{loadError}</p> : null}
      </Card>

      {latest ? (
        <Card>
          <CardTitle>Current Model</CardTitle>
          <div className="metadata-row"><span>{latest.model.model_id}</span><span>Adapter: {latest.model.adapter_id ?? "none"}</span></div>
          <EmotionMeter emotion={latest.emotion} />
        </Card>
      ) : null}

      <Card className="chat-history" aria-label="Chat history">
        {history.length === 0 ? <p className="muted">No messages yet.</p> : null}
        {history.map((turn, index) =>
          turn.role === "system" ? (
            <TicketNotice
              key={`system-${turn.ticketId}`}
              ticketId={turn.ticketId}
              kind={turn.kind}
              initialStatus={turn.status}
            />
          ) : (
            <div key={`${turn.role}-${index}`} className={`chat-bubble ${turn.role}`}>
              <strong>{turn.role === "user" ? "You" : "SUZKA"}</strong>
              <p>{turn.content}</p>
              {turn.failed ? <span className="error">Not sent: {turn.failed}</span> : null}
              {turn.role === "assistant" && turn.result?.transcript_saved === false ? (
                <span className="muted">This turn was not saved to the chat history.</span>
              ) : null}
              {turn.role === "assistant" && turn.result ? (
                <ReviewActions
                  result={turn.result}
                  current={incarnation}
                  disabled={mutation.isPending}
                  onSend={send}
                />
              ) : null}
            </div>
          ),
        )}
        {mutation.isPending ? (
          <div className="chat-bubble assistant pending" role="status" aria-label="SUZKA is responding">
            <strong>SUZKA</strong>
            <span className="spinner" aria-hidden="true" />
            <span className="sr-only">SUZKA is responding…</span>
          </div>
        ) : null}
      </Card>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          submitDraft();
        }}
      >
        <Textarea
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          onKeyDown={(event) => {
            // Ignore Enter while an IME is still composing (e.g. Japanese conversion).
            if (event.nativeEvent.isComposing || event.keyCode === 229) return;
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
              event.preventDefault();
              submitDraft();
            }
          }}
          placeholder="Send a message to PROJECT-SUZKA (Ctrl+Enter to send)"
        />
        <Button disabled={mutation.isPending || !message.trim()} type="submit">{mutation.isPending ? "Sending" : "Send"}</Button>
      </form>

      {mutation.error ? <p className="error">{mutation.error.message}</p> : null}
    </div>
  );
}

function ReviewActions({
  result,
  current,
  disabled,
  onSend,
}: {
  result: ChatResponse;
  current: string | null;
  disabled: boolean;
  onSend: (text: string) => void;
}) {
  const experienceId = result.experience_id;
  // Only turns with an Experience, from a server with an active R16 driver.
  if (!experienceId || !result.incarnation) return null;
  const stale = result.incarnation !== current;
  // A null current incarnation means the driver is now inactive, not that
  // the server restarted (review P2-2).
  const note = current === null ? "Review is unavailable right now." : "Unavailable after a server restart.";
  return (
    <div className="review-actions">
      <Button
        type="button"
        disabled={disabled || stale}
        onClick={() => onSend(reviewCommand("REVIEW_EXPERIENCE", experienceId))}
      >
        Review this
      </Button>
      <Button
        type="button"
        disabled={disabled || stale}
        onClick={() => onSend(reviewCommand("COMMIT_REVIEW_EXPERIENCE", experienceId))}
      >
        Commit to review
      </Button>
      {stale ? <span className="muted">{note}</span> : null}
    </div>
  );
}

function TicketNotice({
  ticketId,
  kind,
  initialStatus,
}: {
  ticketId: string;
  kind: CommandKind;
  initialStatus: string;
}) {
  const [status, setStatus] = useState<CommandStatus>({ status: initialStatus, reason: null });
  const [gaveUp, setGaveUp] = useState(false);

  useEffect(() => {
    // A ticket already settled at submission (e.g. dropped) is not polled.
    if (FINAL.has(initialStatus)) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    // A fixed deadline, so slow requests count toward the 60 s (P2-4).
    const deadline = Date.now() + POLL_LIMIT_MS;
    let delay = FIRST_POLL_MS;
    const poll = () => {
      timer = setTimeout(async () => {
        try {
          const next = await api.commandStatus(ticketId);
          if (cancelled) return;
          setStatus(next);
          if (next.status !== "pending") return;
        } catch (error) {
          if (cancelled) return;
          if (error instanceof ApiError && error.status === 404) {
            // Unknown or evicted: final, not "in progress" (P2-1).
            setStatus({ status: "not_found", reason: null });
            return;
          }
        }
        const remaining = deadline - Date.now();
        if (remaining <= 0) {
          setGaveUp(true);
          return;
        }
        delay = Math.min(delay * 2, remaining);
        poll();
      }, delay);
    };
    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [ticketId, initialStatus]);

  const label = STATUS_LABEL[status.status] ?? status.status;
  return (
    <div className="system-notice" role="status" aria-label="System notice">
      <strong>System</strong>
      <p>
        {KIND_LABEL[kind] ?? "Command"}: {label}
        {status.reason ? ` (${status.reason})` : ""}
        {gaveUp && status.status === "pending" ? " — no result yet; resend the request later." : ""}
      </p>
    </div>
  );
}
