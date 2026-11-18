// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";
import { safeBackendPath, isAllowed } from "@/lib/admin-proxy-path";

const fetchMock = vi.fn();

async function load() {
  vi.resetModules();
  vi.stubEnv("SUZKA_ADMIN_TOKEN", "tok");
  vi.stubEnv("SUZKA_BACKEND_URL", "http://127.0.0.1:8000");
  return import("./[...path]/route");
}

// Simulates what Next hands the handler: raw URL + decoded params.
function call(mod: Awaited<ReturnType<typeof load>>, method: "GET" | "POST", rawPath: string, params: string[], init: RequestInit = {}) {
  const request = new NextRequest(`http://suzka.lan:8080/admin-proxy${rawPath}`, { method, ...init });
  return (method === "GET" ? mod.GET : mod.POST)(request, { params: Promise.resolve({ path: params }) });
}

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }));
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("admin proxy request shape (allowed routes unchanged)", () => {
  it("GET state/emotion relays only Content-Type and the admin token", async () => {
    const mod = await load();
    const res = await call(mod, "GET", "/state/emotion", ["state", "emotion"], {
      headers: { Origin: "http://evil", "X-SUZKA-R18": "1", "X-Forwarded-For": "10.0.0.9", Host: "x", Cookie: "a=b" },
    });
    expect(res.status).toBe(200);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("http://127.0.0.1:8000/api/state/emotion");
    expect(init.method).toBe("GET");
    expect(init.body).toBeUndefined();
    expect(init.headers).toEqual({ "Content-Type": "application/json", "X-SUZKA-Admin-Token": "tok" });
  });

  it("POST adapters/x/approve forwards body, query and content type", async () => {
    const mod = await load();
    await call(mod, "POST", "/adapters/ad1/approve?a=1", ["adapters", "ad1", "approve"], {
      body: '{"k":1}',
      headers: { "Content-Type": "text/plain", Origin: "http://evil" },
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("http://127.0.0.1:8000/api/adapters/ad1/approve?a=1");
    expect(init.body).toBe('{"k":1}');
    expect(init.headers).toEqual({ "Content-Type": "text/plain", "X-SUZKA-Admin-Token": "tok" });
    expect(Object.keys(init.headers).map((k) => k.toLowerCase())).not.toContain("origin");
  });

  it.each([
    ["GET", "/state/emotion"],
    ["GET", "/memory/search"],
    ["GET", "/memory/episodes/e1"],
    ["GET", "/memory/semantic/s1"],
    ["POST", "/chat/debug"],
    ["POST", "/sleep/run"],
    ["GET", "/adapters"],
    ["POST", "/adapters/a/evaluate"],
    ["POST", "/adapters/a/trial"],
    ["POST", "/adapters/a/activate"],
    ["POST", "/adapters/a/reject"],
  ])("%s %s still allowed", async (method, path) => {
    const mod = await load();
    const res = await call(mod, method as "GET" | "POST", path, path.slice(1).split("/"), method === "POST" ? { body: "{}" } : {});
    expect(res.status).toBe(200);
    expect(String(fetchMock.mock.calls[0][0])).toBe(`http://127.0.0.1:8000/api${path}`);
  });
});

describe("admin proxy path hardening matrix", () => {
  // [raw URL path as sent, params as Next might decode them]
  const hostile: Array<[string, string[]]> = [
    ["/r18/approvals", ["r18", "approvals"]],
    ["/R18/approvals", ["R18", "approvals"]],
    ["/adapters/../r18/x", ["adapters", "..", "r18", "x"]],
    ["/adapters/%2e%2e/r18/x", ["adapters", "..", "r18", "x"]],
    ["/adapters/%2E%2E/r18/x", ["adapters", "..", "r18", "x"]],
    ["/adapters/./x/evaluate", ["adapters", ".", "x", "evaluate"]],
    ["/adapters/..%2fr18%2fx/evaluate", ["adapters", "../r18/x", "evaluate"]],
    ["/adapters/..%5cr18/evaluate", ["adapters", "..\\r18", "evaluate"]],
    ["/adapters/a%5c..%5c..%5cr18/evaluate", ["adapters", "a\\..\\..\\r18", "evaluate"]],
    ["/adapters/a\\..\\r18/evaluate", ["adapters", "a\\..\\r18", "evaluate"]],
    ["/adapters/a%23/evaluate", ["adapters", "a#", "evaluate"]],
    ["/adapters/a%2523/evaluate", ["adapters", "a%23", "evaluate"]],
    ["/adapters/a%25/evaluate", ["adapters", "a%", "evaluate"]],
    ["/adapters//evaluate", ["adapters", "", "evaluate"]],
    ["/adapters/", ["adapters", ""]],
    ["/memory/episodes/%2e%2e%2f%2e%2e%2fr18", ["memory", "episodes", "../../r18"]],
    ["/memory/episodes/a%00b", ["memory", "episodes", "a\u0000b"]],
    ["/memory/episodes/a%3fb", ["memory", "episodes", "a?b"]],
    ["/state/emotion", ["state", "emotion", ""]],
  ];

  it.each(hostile)("rejects %s", async (raw, params) => {
    const mod = await load();
    const res = await call(mod, params[0] === "adapters" && params.length > 2 ? "POST" : "GET", raw, params, {});
    expect(res.status).toBe(404);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects when the raw path differs from the decoded params (encoded dots/slashes)", async () => {
    const mod = await load();
    const res = await call(mod, "GET", "/state%2femotion", ["state", "emotion"]);
    expect(res.status).toBe(404);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects missing params", () => {
    expect(safeBackendPath(undefined)).toBeNull();
    expect(safeBackendPath([])).toBeNull();
  });

  it("no allowlist pattern can match anything under r18", () => {
    for (const method of ["GET", "POST"]) {
      for (const p of ["r18", "r18/approvals", "api/r18/x", "adapters/r18/evaluate/../x", "memory/episodes/r18/x"]) {
        expect(safeBackendPath(p.split("/")) === null || !isAllowed(method, p)).toBe(true);
      }
    }
    // an id literally named r18 is also refused (explicit deny)
    expect(safeBackendPath(["adapters", "r18", "evaluate"])).toBeNull();
  });
});
