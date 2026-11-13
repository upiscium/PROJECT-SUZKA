"use client";

import { useMutation } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { ApiError, api, type ChatResponse, type CommandKind, type CommandStatus } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { EmotionMeter } from "@/components/emotion-meter";

type ChatTurn =
  | { role: "user" | "assistant"; content: string; result?: ChatResponse }
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

export function reviewCommand(kind: CommandKind, experienceId: string): string {
  // The screen builds the exact command text the owners check (DRV-3).
  return kind === "REVIEW_EXPERIENCE"
    ? `REVIEW_EXPERIENCE:${experienceId}`
    : `COMMIT_REVIEW_EXPERIENCE:1:${experienceId}`;
}

export function ChatClient() {
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const mutation = useMutation({
    mutationFn: api.chat,
    onSuccess: (result, variables) => {
      const ticket = result.command_ticket;
      setHistory((current) => [
        ...current,
        { role: "user", content: variables.message },
        { role: "assistant", content: result.response, result },
        ...(ticket
          ? [{ role: "system" as const, ticketId: ticket.ticket_id, kind: ticket.kind, status: ticket.status }]
          : []),
      ]);
      setMessage("");
    },
  });

  const latest = [...history]
    .reverse()
    .find((turn): turn is Extract<ChatTurn, { result?: ChatResponse }> => "result" in turn && !!turn.result)?.result;
  // The server incarnation of the latest response: a different value means
  // the server restarted, so earlier turns can no longer be reviewed.
  const incarnation = latest?.incarnation ?? null;

  const send = (text: string) => mutation.mutate({ message: text, attachments: [], debug: false });

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1 className="page-title">Chat</h1>
          <p className="page-subtitle">Normal chat shows only user-facing answers and emotion state.</p>
        </div>
        {latest ? <Badge>{latest.model.model_id}</Badge> : null}
      </header>

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
      </Card>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          if (message.trim()) send(message);
        }}
      >
        <Textarea value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Send a message to PROJECT-SUZKA" />
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
