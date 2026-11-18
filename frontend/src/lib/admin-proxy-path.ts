// Path hardening for the admin proxy (IP-17c, D1 §4). The proxy is a
// confused-deputy relay: it must never be steerable to /api/r18/*.
export const ALLOWED_ROUTES = [
  { method: "GET", pattern: /^state\/emotion$/ },
  { method: "GET", pattern: /^memory\/search$/ },
  { method: "GET", pattern: /^memory\/episodes\/[^/]+$/ },
  { method: "GET", pattern: /^memory\/semantic\/[^/]+$/ },
  { method: "POST", pattern: /^chat\/debug$/ },
  { method: "POST", pattern: /^sleep\/run$/ },
  { method: "GET", pattern: /^adapters$/ },
  { method: "POST", pattern: /^adapters\/[^/]+\/(evaluate|trial|approve|activate|reject)$/ },
];

const FORBIDDEN_CHARS = /[\\#%?\u0000-\u001f\u007f]/;

/** Returns the safe backend path, or null if the requested path is suspicious. */
export function safeBackendPath(segments: string[] | undefined, rawPathname?: string): string | null {
  if (!segments || segments.length === 0) return null;
  for (const segment of segments) {
    if (segment === "" || segment === "." || segment === "..") return null;
    if (segment.includes("/") || FORBIDDEN_CHARS.test(segment)) return null;
  }
  // Anything resolving under r18 is denied explicitly, regardless of allowlist.
  if (segments.some((segment) => segment.toLowerCase() === "r18")) return null;
  const joined = segments.join("/");
  if (rawPathname !== undefined) {
    // Raw request path must be exactly the prefix + the decoded path: no
    // percent-encoding, backslash or dot-segment tricks survive to here.
    if (rawPathname !== `/admin-proxy/${joined}`) return null;
  }
  return joined;
}

export function isAllowed(method: string, path: string): boolean {
  return ALLOWED_ROUTES.some((route) => route.method === method && route.pattern.test(path));
}
