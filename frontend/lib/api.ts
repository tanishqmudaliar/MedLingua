const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type User = { id: string; username: string; name: string; email: string };
export type DocumentTranslationCache = { summary: string | null; extracted_text: string | null; extracted_chunks: string[] };
export type Document = { id: string; original_filename: string; media_type: string; extracted_text: string; summary: string | null; translations: Partial<Record<TranslationLanguage, DocumentTranslationCache>>; created_at: string };
export type TranslationLanguage = "hi" | "mr" | "ta";
export type DocumentTranslation = { language: TranslationLanguage; summary: string | null; extracted_text: string | null; extracted_chunks: string[] };
export type TranslationEvent =
  | { type: "status"; stage: string; message: string }
  | { type: "summary"; text: string }
  | { type: "text_progress"; completed: number; total: number; chunk: string }
  | { type: "complete"; extracted_text: string }
  | { type: "error"; message: string; status: number };

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, credentials: "include", headers });
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(`Cannot connect to the MedLingua backend at ${API_URL}. Make sure it is running and try again.`);
    }
    throw error;
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail ?? "Request failed", response.status);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

export const api = {
  me: () => request<User>("/auth/me"),
  signup: (data: { username: string; name: string; email: string; password: string }) => request<User>("/auth/signup", { method: "POST", body: JSON.stringify(data) }),
  login: (data: { username: string; password: string }) => request<User>("/auth/login", { method: "POST", body: JSON.stringify(data) }),
  logout: () => request<void>("/auth/logout", { method: "POST" }),
  documents: () => request<Document[]>("/documents"),
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  translateDocument: async (
    id: string,
    language: TranslationLanguage,
    onEvent: (event: TranslationEvent) => void,
  ) => {
    let response: Response;
    try {
      response = await fetch(`${API_URL}/documents/${id}/translate`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language }),
      });
    } catch (error) {
      if (error instanceof TypeError) {
        throw new Error(`Cannot connect to the MedLingua backend at ${API_URL}. Make sure it is running and try again.`);
      }
      throw error;
    }
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new ApiError(body.detail ?? "Translation request failed", response.status);
    }
    if (!response.body) throw new Error("The backend did not provide a translation stream.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    while (true) {
      const { done, value } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      const lines = pending.split("\n");
      pending = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line) as TranslationEvent;
        if (event.type === "error") throw new ApiError(event.message, event.status);
        onEvent(event);
      }
      if (done) break;
    }
    if (pending.trim()) {
      const event = JSON.parse(pending) as TranslationEvent;
      if (event.type === "error") throw new ApiError(event.message, event.status);
      onEvent(event);
    }
  },
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<Document>("/documents", { method: "POST", body });
  }
};
