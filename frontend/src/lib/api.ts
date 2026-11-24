export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const ADMIN_PROXY_BASE_URL = "/admin-proxy";

export type Emotion = { valence: number; arousal: number; optimal_loss: number };
export type ModelInfo = { model_id: string; adapter_id: string | null };
export type Attachment = { type: string; url?: string; name?: string; content_type?: string };

// chat_session_id names a display-only transcript session; it never selects a Context.
export type ChatRequest = { message: string; attachments?: Attachment[]; debug?: boolean; chat_session_id?: string };
export type ChatSession = { session_id: string; title: string; created_at: string; updated_at: string };
export type ChatLogTurn = { role: "user" | "assistant"; content: string; episode_id: string | null; created_at: string };
export type ChatSessionDetail = { session: ChatSession; turns: ChatLogTurn[] };
export type CommandKind = "REVIEW_EXPERIENCE" | "COMMIT_REVIEW_EXPERIENCE";
export type CommandTicket = { ticket_id: string; kind: CommandKind; status: string };
export type CommandStatus = { status: string; reason: string | null };

export type ChatResponse = {
  episode_id: string;
  response: string;
  emotion: Emotion;
  model: ModelInfo;
  // IP-6b/IP-7 (#370 DRV-3, D1 2-1): present when the R16 driver is active.
  experience_id?: string | null;
  command_ticket?: CommandTicket | null;
  incarnation?: string | null;
  // Whether the turn reached the chat_session_id transcript; null without one.
  transcript_saved?: boolean | null;
};

export type RetrievedMemory = {
  db1_results: Array<{ id: string; user_input: string; response: string; record_type: string }>;
  db2_results: Array<{ id: string; text: string; record_type: string }>;
};

export type DebugChatResponse = ChatResponse & {
  hidden_thought: string;
  loss: number;
  prompt: string;
  retrieved_memory: RetrievedMemory;
  generation_params: { max_new_tokens: number; temperature: number; top_p: number; do_sample: boolean };
};

export type EpisodeMemory = {
  id: string;
  user_input: string;
  response: string;
  loss: number;
  emotion_valence: number;
  emotion_arousal: number;
  record_type: string;
  archived: boolean;
  created_at: string;
};

export type SemanticMemory = {
  id: string;
  text: string;
  source_episode_ids: string[];
  record_type: string;
  created_at: string;
};

export type MemorySearchResponse = { db1_results: EpisodeMemory[]; db2_results: SemanticMemory[] };

export type Adapter = {
  adapter_id: string;
  base_model: string;
  path: string;
  status: string;
  dataset_path: string;
  dataset_hash: string;
  eval_score: number | null;
  eval_result_path: string | null;
  created_at: string;
  updated_at: string;
  notes: string;
};

export type AdapterListResponse = { adapters: Adapter[] };
export type AdapterEvaluateResponse = { adapter_id: string; score: number; decision: string; result_path: string; status: string };
export type SleepRunResponse = {
  selected_episode_ids: string[];
  semantic_memory_ids: string[];
  dream_dataset_path: string | null;
  adapter_id: string | null;
  adapter_status: string | null;
  dry_run: boolean | null;
};

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// The API's guard refusal is {"status":"refused","reason":...,"hint":...}; show the
// hint in plain words when present, otherwise keep the generic text.
export function describeApiError(status: number, statusText: string, detail: string): string {
  try {
    const body: unknown = JSON.parse(detail);
    if (body !== null && typeof body === "object") {
      const { status: kind, hint } = body as { status?: unknown; hint?: unknown };
      if (kind === "refused" && typeof hint === "string" && hint !== "") {
        return `${status} The API refused this request. To fix: ${hint}.`;
      }
    }
  } catch {
    // not JSON: fall through to the generic text
  }
  return `${status} ${statusText}: ${detail}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return requestUrl<T>(`${API_BASE_URL}${path}`, init);
}

async function adminRequest<T>(path: string, init?: RequestInit): Promise<T> {
  return requestUrl<T>(`${ADMIN_PROXY_BASE_URL}${path}`, init);
}

async function requestUrl<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new ApiError(response.status, describeApiError(response.status, response.statusText, detail));
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  chat: (body: ChatRequest) => request<ChatResponse>("/api/chat", { method: "POST", body: JSON.stringify(body) }),
  chatSessions: () => request<{ sessions: ChatSession[] }>("/api/chat/sessions"),
  createChatSession: () => request<ChatSession>("/api/chat/sessions", { method: "POST", body: JSON.stringify({}) }),
  chatSession: (sessionId: string) => request<ChatSessionDetail>(`/api/chat/sessions/${encodeURIComponent(sessionId)}`),
  deleteChatSession: (sessionId: string) => request<void>(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" }),
  commandStatus: (ticketId: string) => request<CommandStatus>(`/api/chat/commands/${encodeURIComponent(ticketId)}`),
  debugChat: (body: ChatRequest) => adminRequest<DebugChatResponse>("/chat/debug", { method: "POST", body: JSON.stringify(body) }),
  emotion: () => adminRequest<Emotion>("/state/emotion"),
  memorySearch: (query: string) => adminRequest<MemorySearchResponse>(`/memory/search?query=${encodeURIComponent(query)}`),
  sleepRun: () => adminRequest<SleepRunResponse>("/sleep/run", { method: "POST" }),
  adapters: () => adminRequest<AdapterListResponse>("/adapters"),
  evaluateAdapter: (adapterId: string, deterministic_score?: number) => adminRequest<AdapterEvaluateResponse>(`/adapters/${adapterId}/evaluate`, { method: "POST", body: JSON.stringify({ deterministic_score }) }),
  trialAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/trial`, { method: "POST" }),
  approveAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/approve`, { method: "POST" }),
  activateAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/activate`, { method: "POST" }),
  rejectAdapter: (adapterId: string) => adminRequest<Adapter>(`/adapters/${adapterId}/reject`, { method: "POST" }),
};
