"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { api, ApiError, Document, User } from "../lib/api";

export default function Home() {
  const [user, setUser] = useState<User | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const uploadFormRef = useRef<HTMLFormElement>(null);

  useEffect(() => { api.me().then(setUser).catch(() => undefined); }, []);
  useEffect(() => {
    if (user) {
      api.documents().then((items) => {
        setDocuments(items);
        setSelectedId(items[0]?.id ?? null);
      }).catch((e) => setError(e.message));
    }
  }, [user]);

  async function submitAuth(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(""); setBusy(true);
    const data = Object.fromEntries(new FormData(event.currentTarget));
    try {
      const nextUser = mode === "login" ? await api.login({ username: String(data.username), password: String(data.password) }) : await api.signup({ username: String(data.username), name: String(data.name), email: String(data.email), password: String(data.password) });
      setUser(nextUser);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        setMode("signup");
        setError("Account not found. Sign up to create your account.");
      } else {
        setError(e instanceof Error ? e.message : "Unable to authenticate");
      }
    } finally { setBusy(false); }
  }

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError("");
    const file = (event.currentTarget.elements.namedItem("file") as HTMLInputElement).files?.[0];
    if (!file) return;
    setBusy(true);
    try {
      const document = await api.upload(file);
      setDocuments((current) => [document, ...current]);
      setSelectedId(document.id);
      uploadFormRef.current?.reset();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function removeDocument(document: Document) {
    if (!window.confirm(`Delete "${document.original_filename}"? This cannot be undone.`)) return;
    setError(""); setBusy(true);
    try {
      await api.deleteDocument(document.id);
      const remaining = documents.filter((item) => item.id !== document.id);
      setDocuments(remaining);
      setSelectedId(remaining[0]?.id ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete document");
    } finally {
      setBusy(false);
    }
  }

  if (!user) return <main><h1>MedLingua</h1><p>{mode === "login" ? "Log in to extract text." : "Create your account."}</p><form onSubmit={submitAuth}>
    <input name="username" placeholder="Username" required minLength={3} />
    {mode === "signup" && <><input name="name" placeholder="Full name" required /><input name="email" type="email" placeholder="Email" required /></>}
    <input name="password" type="password" placeholder="Password" required minLength={8} />
    <button disabled={busy}>{busy ? "Please wait..." : mode === "login" ? "Log in" : "Sign up"}</button>
    <button type="button" onClick={() => setMode(mode === "login" ? "signup" : "login")}>{mode === "login" ? "Need an account?" : "Already have an account?"}</button>
  </form>{error && <p className="error">{error}</p>}</main>;

  const selectedDocument = documents.find((document) => document.id === selectedId) ?? null;
  return <main className="app-shell"><header className="row"><h1>Welcome, {user.name}</h1><button onClick={() => api.logout().then(() => setUser(null))}>Log out</button></header>
    <div className="workspace">
      <aside className="sidebar"><h2>Uploads</h2>{documents.length === 0 ? <p>No uploads yet.</p> : <nav aria-label="Uploaded media">{documents.map((document) => <button className={`document-link ${document.id === selectedId ? "selected" : ""}`} key={document.id} onClick={() => setSelectedId(document.id)}>{document.original_filename}</button>)}</nav>}</aside>
      <section className="content-panel">
        <section className="card"><h2>Extract text</h2><form ref={uploadFormRef} onSubmit={upload}><input name="file" type="file" accept=".pdf,image/png,image/jpeg,image/webp,image/bmp,image/tiff" required /><button disabled={busy}>{busy ? "Working..." : "Upload and extract"}</button></form></section>
        {error && <p className="error">{error}</p>}
        {selectedDocument ? <article className="card"><div className="row"><div><h2>{selectedDocument.original_filename}</h2><small>{new Date(selectedDocument.created_at).toLocaleString()}</small></div><button className="danger" disabled={busy} onClick={() => removeDocument(selectedDocument)}>Delete record</button></div><pre>{selectedDocument.extracted_text || "No text was detected."}</pre></article> : <section className="card"><p>Upload a PDF or image to view its extracted text here.</p></section>}
      </section>
    </div>
  </main>;
}
