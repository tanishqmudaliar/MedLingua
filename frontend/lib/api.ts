const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type User = { id: string; username: string; name: string; email: string };
export type DocumentTranslationCache = {
  summary: string | null;
  extracted_text: string | null;
  extracted_chunks: string[];
};
export type TranslationLanguage = "hi" | "mr" | "ta";
export type Document = {
  id: string;
  original_filename: string;
  media_type: string;
  extracted_text: string;
  summary: string | null;
  summary_method?: "llm" | "extractive_fallback" | null;
  processing_status: "uploaded" | "extracting" | "summarizing" | "summarized" | "translating" | "completed" | "failed";
  error_message?: string | null;
  hindi_summary?: string | null;
  marathi_summary?: string | null;
  tamil_summary?: string | null;
  translations_status?: Record<TranslationLanguage, "pending" | "translating" | "completed" | "failed">;
  translations: Partial<Record<TranslationLanguage, DocumentTranslationCache>>;
  created_at: string;
};

export type DocumentTranslation = {
  language: TranslationLanguage;
  summary: string | null;
  extracted_text: string | null;
  extracted_chunks: string[];
};

export type TranslationEvent =
  | { type: "status"; stage: string; message: string }
  | { type: "summary"; text: string }
  | { type: "text_progress"; completed: number; total: number; chunk: string }
  | { type: "complete"; extracted_text: string }
  | { type: "error"; message: string; status: number };

export type UploadProgressEvent = {
  type: "stage" | "complete" | "error";
  stage?: "uploading" | "extracting" | "summarizing" | "saving" | "completed" | "failed";
  message?: string;
  percent?: number;
  document?: Document;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

export type Conversation = {
  id: string;
  document_id: string;
  messages: ChatMessage[];
};

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
  signup: (data: { username: string; name: string; email: string; password: string }) =>
    request<User>("/auth/signup", { method: "POST", body: JSON.stringify(data) }),
  login: (data: { username: string; password: string }) =>
    request<User>("/auth/login", { method: "POST", body: JSON.stringify(data) }),
  logout: () => request<void>("/auth/logout", { method: "POST" }),
  documents: () => request<Document[]>("/documents"),
  document: (id: string) => request<Document>(`/documents/${id}`),
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  retranslate: (id: string, language: TranslationLanguage) =>
    request<Document>(`/documents/${id}/retranslate`, {
      method: "POST",
      body: JSON.stringify({ language }),
    }),
  getConversation: (id: string) => request<Conversation>(`/documents/${id}/conversation`),
  chatStream: async (
    id: string,
    message: string,
    onToken: (token: string) => void,
    onComplete: (fullContent: string) => void,
  ) => {
    let response: Response;
    try {
      response = await fetch(`${API_URL}/documents/${id}/chat`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
    } catch (error) {
      if (error instanceof TypeError) {
        throw new Error(`Cannot connect to the MedLingua backend at ${API_URL}.`);
      }
      throw error;
    }
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new ApiError(body.detail ?? "Chat request failed", response.status);
    }
    if (!response.body) throw new Error("No response stream received from backend.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    let accumulated = "";

    while (true) {
      const { done, value } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      const lines = pending.split("\n");
      pending = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const event = JSON.parse(line);
          if (event.type === "token") {
            accumulated += event.content;
            onToken(event.content);
          } else if (event.type === "complete") {
            onComplete(event.content || accumulated);
          } else if (event.type === "error") {
            throw new Error(event.message);
          }
        } catch (e) {
          if (e instanceof Error && e.message !== "Unexpected end of JSON input") {
            throw e;
          }
        }
      }
      if (done) break;
    }
    onComplete(accumulated);
  },
  translateDocument: async (
    id: string,
    language: TranslationLanguage,
    onEvent: (event: TranslationEvent) => void,
    force: boolean = false,
  ) => {
    let response: Response;
    try {
      response = await fetch(`${API_URL}/documents/${id}/translate`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language, force }),
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
  upload: async (
    file: File,
    onProgress?: (event: UploadProgressEvent) => void,
  ): Promise<Document> => {
    const body = new FormData();
    body.append("file", file);
    let response: Response;
    try {
      response = await fetch(`${API_URL}/documents`, {
        method: "POST",
        credentials: "include",
        headers: { Accept: "application/x-ndjson" },
        body,
      });
    } catch (error) {
      if (error instanceof TypeError) {
        throw new Error(`Cannot connect to the MedLingua backend at ${API_URL}. Make sure it is running and try again.`);
      }
      throw error;
    }
    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new ApiError(errBody.detail ?? "Upload failed", response.status);
    }

    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/x-ndjson") && response.body) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      let completedDoc: Document | null = null;

      while (true) {
        const { done, value } = await reader.read();
        pending += decoder.decode(value, { stream: !done });
        const lines = pending.split("\n");
        pending = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const event = JSON.parse(line) as UploadProgressEvent;
            if (event.type === "error") {
              throw new ApiError(event.message || "Upload failed", 500);
            }
            if (event.type === "complete" && event.document) {
              completedDoc = event.document;
            }
            if (onProgress) {
              onProgress(event);
            }
          } catch (e) {
            if (e instanceof ApiError) throw e;
          }
        }
        if (done) break;
      }
      if (completedDoc) return completedDoc;
    }
    return response.json();
  },
  getHardwareStatus: () => request<any>("/system/hardware"),
};
