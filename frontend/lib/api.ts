const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type User = { id: string; username: string; name: string; email: string };
export type Document = { id: string; original_filename: string; media_type: string; extracted_text: string; created_at: string };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const response = await fetch(`${API_URL}${path}`, { ...init, credentials: "include", headers });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? "Request failed");
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
