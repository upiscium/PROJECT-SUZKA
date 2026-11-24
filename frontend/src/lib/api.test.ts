import { describe, expect, it, vi, beforeEach } from "vitest";
import { api, ApiError, describeApiError } from "./api";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

function jsonResponse(body: unknown) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(body) });
}

describe("api client", () => {
  it("/chat sends requests to /api/chat", async () => {
    fetchMock.mockReturnValue(jsonResponse({ episode_id: "e", response: "ok", emotion: { valence: 0, arousal: 0, optimal_loss: 1 }, model: { model_id: "m", adapter_id: null } }));

    await api.chat({ message: "hello", attachments: [], debug: false });

    expect(fetchMock).toHaveBeenCalledWith("http://127.0.0.1:8000/api/chat", expect.objectContaining({ method: "POST" }));
  });

  it("/debug sends requests through the server-side admin proxy", async () => {
    fetchMock.mockReturnValue(jsonResponse({}));

    await api.debugChat({ message: "hello", attachments: [], debug: true });

    expect(fetchMock).toHaveBeenCalledWith("/admin-proxy/chat/debug", expect.objectContaining({ method: "POST" }));
    expect(fetchMock.mock.calls[0][1]?.headers).not.toHaveProperty("X-SUZKA-Admin-Token");
  });

  it("adapter actions call backend lifecycle endpoints", async () => {
    fetchMock.mockReturnValue(jsonResponse({}));

    await api.evaluateAdapter("a");
    await api.trialAdapter("a");
    await api.approveAdapter("a");
    await api.activateAdapter("a");
    await api.rejectAdapter("a");

    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      "/admin-proxy/adapters/a/evaluate",
      "/admin-proxy/adapters/a/trial",
      "/admin-proxy/adapters/a/approve",
      "/admin-proxy/adapters/a/activate",
      "/admin-proxy/adapters/a/reject",
    ]);
  });

  it("sleep page action calls /api/sleep/run", async () => {
    fetchMock.mockReturnValue(jsonResponse({}));

    await api.sleepRun();

    expect(fetchMock).toHaveBeenCalledWith("/admin-proxy/sleep/run", expect.objectContaining({ method: "POST" }));
  });
});

describe("API error text", () => {
  it("shows the hint of a guard refusal in plain words", async () => {
    const detail = '{"status":"refused","reason":"api_host_not_accepted","hint":"add this host name to api.allowed_hosts"}';
    fetchMock.mockReturnValue(Promise.resolve({ ok: false, status: 403, statusText: "Forbidden", text: () => Promise.resolve(detail) }));

    const error = await api.chatSessions().catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toBe("403 The API refused this request. To fix: add this host name to api.allowed_hosts.");
  });

  it("keeps the existing text for other errors", () => {
    expect(describeApiError(500, "Server Error", "boom")).toBe("500 Server Error: boom");
    expect(describeApiError(403, "Forbidden", '{"status":"refused","reason":"x"}')).toBe('403 Forbidden: {"status":"refused","reason":"x"}');
    expect(describeApiError(400, "Bad", '{"hint":"h"}')).toBe('400 Bad: {"hint":"h"}');
    expect(describeApiError(400, "Bad", "null")).toBe("400 Bad: null");
  });
});
