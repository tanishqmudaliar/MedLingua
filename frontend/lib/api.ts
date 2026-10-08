const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type User = { id: string; username: string; name: string; email: string };
export type Document = { id: string; original_filename: string; media_type: string; extracted_text: string; summary: string | null; created_at: string };

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
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<Document>("/documents", { method: "POST", body });
  }
};
