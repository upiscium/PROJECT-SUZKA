import { NextRequest, NextResponse } from "next/server";
import { isAllowed, safeBackendPath } from "@/lib/admin-proxy-path";

const BACKEND_URL = process.env.SUZKA_BACKEND_URL ?? "http://127.0.0.1:8000";

type RouteContext = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, context: RouteContext) {
  return proxyAdminRequest(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxyAdminRequest(request, context);
}

async function proxyAdminRequest(request: NextRequest, context: RouteContext): Promise<NextResponse> {
  const { path } = await context.params;
  const backendPath = safeBackendPath(path, request.nextUrl.pathname);
  if (backendPath === null || !isAllowed(request.method, backendPath)) {
    return NextResponse.json({ detail: "Admin proxy route is not allowed" }, { status: 404 });
  }

  const target = new URL(`/api/${backendPath}`, BACKEND_URL);
  target.search = request.nextUrl.search;
  const body = request.method === "GET" ? undefined : await request.text();
  const response = await fetch(target, {
    method: request.method,
    headers: {
      "Content-Type": request.headers.get("Content-Type") ?? "application/json",
    },
    body,
    redirect: "manual",
  });

  return new NextResponse(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: { "Content-Type": response.headers.get("Content-Type") ?? "application/json" },
  });
}

