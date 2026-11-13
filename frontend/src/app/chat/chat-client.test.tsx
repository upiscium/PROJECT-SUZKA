import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatClient, FIRST_POLL_MS, POLL_LIMIT_MS, reviewCommand } from "./chat-client";

const fetchMock = vi.fn();
const EXPERIENCE = "experience-00000000-0000-4000-8000-000000000001";

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.useRealTimers();
});

function renderWithQuery() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><ChatClient /></QueryClientProvider>);
}

function chatResponse(extra: Record<string, unknown> = {}) {
  return {
    ok: true,
    json: async () => ({
      episode_id: "episode-1",
      response: "Visible answer",
      emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
      model: { model_id: "google/gemma-4-E4B", adapter_id: null },
      ...extra,
    }),
  };
}

function statusResponse(status: string, reason: string | null = null) {
  return { ok: true, json: async () => ({ status, reason }) };
}

async function send(text: string) {
  const user = vi.isFakeTimers()
    ? userEvent.setup({ advanceTimers: vi.advanceTimersByTime.bind(vi) })
    : userEvent.setup();
  await user.type(screen.getByPlaceholderText("Send a message to PROJECT-SUZKA"), text);
  await user.click(screen.getByRole("button", { name: "Send" }));
  return user;
}

function sentBodies(): string[] {
  return fetchMock.mock.calls
    .filter(([url]) => String(url).endsWith("/api/chat"))
    .map(([, init]) => JSON.parse((init as RequestInit).body as string).message as string);
}

describe("ChatClient", () => {
  it("renders normal chat response without hidden thought fields", async () => {
    fetchMock.mockResolvedValue(chatResponse());
    renderWithQuery();
    await send("hello");

    expect(await screen.findByText("Visible answer")).toBeInTheDocument();
    expect(screen.queryByText(/hidden_thought/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/raw prompt/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/retrieved memory/i)).not.toBeInTheDocument();
    // No driver: no review actions.
    expect(screen.queryByRole("button", { name: "Review this" })).not.toBeInTheDocument();
  });

  it("builds the exact command text from a turn's experience", async () => {
    expect(reviewCommand("REVIEW_EXPERIENCE", EXPERIENCE)).toBe(`REVIEW_EXPERIENCE:${EXPERIENCE}`);
    expect(reviewCommand("COMMIT_REVIEW_EXPERIENCE", EXPERIENCE)).toBe(
      `COMMIT_REVIEW_EXPERIENCE:1:${EXPERIENCE}`,
    );
    fetchMock.mockResolvedValue(chatResponse({ experience_id: EXPERIENCE, incarnation: "inc-1" }));
    renderWithQuery();
    const user = await send("hello");
    await user.click(await screen.findByRole("button", { name: "Review this" }));
    await waitFor(() => expect(sentBodies()).toContain(`REVIEW_EXPERIENCE:${EXPERIENCE}`));
    const commit = (await screen.findAllByRole("button", { name: "Commit to review" }))[0];
    await user.click(commit);
    await waitFor(() => expect(sentBodies()).toContain(`COMMIT_REVIEW_EXPERIENCE:1:${EXPERIENCE}`));
  });

  it("shows a ticket as a system notice and polls until it settles", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    fetchMock.mockImplementation(async (url: string) => {
      if (String(url).endsWith("/api/chat")) {
        return chatResponse({
          experience_id: EXPERIENCE,
          incarnation: "inc-1",
          command_ticket: { ticket_id: "inc-1.tok", kind: "REVIEW_EXPERIENCE", status: "pending" },
        });
      }
      const polls = fetchMock.mock.calls.filter(([u]) => String(u).includes("/api/chat/commands/")).length;
      return polls < 3 ? statusResponse("pending") : statusResponse("hold", "goal_already_exists");
    });
    renderWithQuery();
    await send(`REVIEW_EXPERIENCE:${EXPERIENCE}`);
    const notice = await screen.findByRole("status", { name: "System notice" });
    expect(notice).toHaveTextContent("Review request: in progress");
    // A system notice is not a chat bubble of the subject.
    expect(notice.className).toContain("system-notice");
    expect(notice.className).not.toContain("chat-bubble");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(FIRST_POLL_MS + 2 * FIRST_POLL_MS + 4 * FIRST_POLL_MS);
    });
    await waitFor(() => expect(notice).toHaveTextContent("Review request: on hold (goal_already_exists)"));
    const pollUrls = fetchMock.mock.calls.map(([u]) => String(u)).filter((u) => u.includes("/commands/"));
    expect(pollUrls[0]).toContain(`/api/chat/commands/${encodeURIComponent("inc-1.tok")}`);
    const settled = pollUrls.length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_LIMIT_MS);
    });
    expect(fetchMock.mock.calls.filter(([u]) => String(u).includes("/commands/")).length).toBe(settled);
  });

  it("stops polling after 60 seconds", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    fetchMock.mockImplementation(async (url: string) =>
      String(url).endsWith("/api/chat")
        ? chatResponse({
            incarnation: "inc-1",
            command_ticket: { ticket_id: "inc-1.slow", kind: "COMMIT_REVIEW_EXPERIENCE", status: "pending" },
          })
        : statusResponse("pending"),
    );
    renderWithQuery();
    await send(`COMMIT_REVIEW_EXPERIENCE:1:${EXPERIENCE}`);
    const notice = await screen.findByRole("status", { name: "System notice" });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_LIMIT_MS + 5_000);
    });
    await waitFor(() => expect(notice).toHaveTextContent("no result yet"));
    const polls = fetchMock.mock.calls.filter(([u]) => String(u).includes("/commands/")).length;
    // 1 + 2 + 4 + 8 + 16 + 29 s: six polls within the 60 s limit.
    expect(polls).toBe(6);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_LIMIT_MS);
    });
    expect(fetchMock.mock.calls.filter(([u]) => String(u).includes("/commands/")).length).toBe(polls);
  });

  it("disables review actions on turns from before a server restart", async () => {
    let incarnation = "inc-1";
    fetchMock.mockImplementation(async () =>
      chatResponse({ experience_id: EXPERIENCE, incarnation }),
    );
    renderWithQuery();
    await send("before restart");
    expect(await screen.findByRole("button", { name: "Review this" })).toBeEnabled();
    incarnation = "inc-2";
    await send("after restart");
    await waitFor(() => expect(screen.getAllByRole("button", { name: "Review this" })).toHaveLength(2));
    const [old, fresh] = screen.getAllByRole("button", { name: "Review this" });
    expect(old).toBeDisabled();
    expect(fresh).toBeEnabled();
    expect(screen.getByText("Unavailable after a server restart.")).toBeInTheDocument();
  });
});
