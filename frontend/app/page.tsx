"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import {
  api,
  ApiError,
  Document,
  DocumentTranslation,
  TranslationLanguage,
  User,
} from "../lib/api";

const LANGUAGES: { code: "en" | TranslationLanguage; label: string }[] = [
  { code: "en", label: "English" },
  { code: "hi", label: "हिन्दी" },
  { code: "mr", label: "मराठी" },
  { code: "ta", label: "தமிழ்" },
];

export default function Home() {
  const [user, setUser] = useState<User | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [activeLanguage, setActiveLanguage] = useState<"en" | TranslationLanguage>("en");
  const [translations, setTranslations] = useState<Record<string, DocumentTranslation>>({});
  const [translationLoadingKey, setTranslationLoadingKey] = useState<string | null>(null);
  const [translationError, setTranslationError] = useState("");
  const [translationStatus, setTranslationStatus] = useState("");
  const [translationProgress, setTranslationProgress] = useState<{ completed: number; total: number } | null>(null);
  const uploadFormRef = useRef<HTMLFormElement>(null);
  const translationSessionRef = useRef(0);

  useEffect(() => { api.me().then(setUser).catch(() => undefined); }, []);
  useEffect(() => {
    if (user) {
      api.documents().then((items) => {
        setDocuments(items);
        setSelectedId(items[0]?.id ?? null);
        setTranslations(Object.fromEntries(items.flatMap((item) =>
          Object.entries(item.translations ?? {}).map(([language, translation]) => [
            `${item.id}:${language}`,
            {
              language: language as TranslationLanguage,
              summary: translation.summary,
              extracted_text: translation.extracted_text,
              extracted_chunks: translation.extracted_chunks,
            },
          ]),
        )));
      }).catch((e) => setError(e.message));
    }
  }, [user]);
  useEffect(() => {
    setActiveLanguage("en");
    setTranslationError("");
    setTranslationStatus("");
    setTranslationProgress(null);
    setTranslationLoadingKey(null);
  }, [selectedId]);

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
      translationSessionRef.current += 1;
      setTranslationLoadingKey(null);
      setTranslationError("");
      setTranslationStatus("");
      setTranslationProgress(null);
      await api.deleteDocument(document.id);
      const remaining = documents.filter((item) => item.id !== document.id);
      setDocuments(remaining);
      setSelectedId(remaining[0]?.id ?? null);
      setTranslations((current) => Object.fromEntries(
        Object.entries(current).filter(([key]) => !key.startsWith(`${document.id}:`)),
      ));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete document");
    } finally {
      setBusy(false);
    }
  }

  async function logout() {
    await api.logout();
    translationSessionRef.current += 1;
    setUser(null);
    setTranslations({});
    setActiveLanguage("en");
    setTranslationLoadingKey(null);
    setTranslationError("");
    setTranslationStatus("");
    setTranslationProgress(null);
  }

  if (!user) return <main><h1>MedLingua</h1><p>{mode === "login" ? "Log in to extract text." : "Create your account."}</p><form onSubmit={submitAuth}>
    <input name="username" placeholder="Username" required minLength={3} />
    {mode === "signup" && <><input name="name" placeholder="Full name" required /><input name="email" type="email" placeholder="Email" required /></>}
    <input name="password" type="password" placeholder="Password" required minLength={8} />
    <button disabled={busy}>{busy ? "Please wait..." : mode === "login" ? "Log in" : "Sign up"}</button>
    <button type="button" onClick={() => setMode(mode === "login" ? "signup" : "login")}>{mode === "login" ? "Need an account?" : "Already have an account?"}</button>
  </form>{error && <p className="error">{error}</p>}</main>;

  const selectedDocument = documents.find((document) => document.id === selectedId) ?? null;
  const translationKey = selectedDocument && activeLanguage !== "en"
    ? `${selectedDocument.id}:${activeLanguage}`
    : null;
  const activeTranslation = translationKey ? translations[translationKey] : null;

  async function changeLanguage(language: "en" | TranslationLanguage) {
    setActiveLanguage(language);
    setTranslationError("");
    if (!selectedDocument || language === "en") return;

    const key = `${selectedDocument.id}:${language}`;
    const cachedTranslation = translations[key];
    if (
      (cachedTranslation?.summary !== null && cachedTranslation?.summary !== undefined && cachedTranslation.extracted_text !== null)
      || translationLoadingKey === key
    ) return;

    const sessionId = translationSessionRef.current;
    setTranslationLoadingKey(key);
    setTranslationStatus("Starting local translation...");
    setTranslationProgress(null);
    setTranslations((current) => ({
      ...current,
      [key]: {
        language,
        summary: cachedTranslation?.summary ?? null,
        extracted_text: cachedTranslation?.extracted_text ?? null,
        extracted_chunks: cachedTranslation?.extracted_chunks ?? [],
      },
    }));
    try {
      await api.translateDocument(selectedDocument.id, language, (event) => {
        if (translationSessionRef.current !== sessionId) return;
        if (event.type === "status") {
          setTranslationStatus(event.message);
        } else if (event.type === "summary") {
          setTranslations((current) => ({
            ...current,
            [key]: { ...current[key], language, summary: event.text, extracted_text: current[key]?.extracted_text ?? null, extracted_chunks: current[key]?.extracted_chunks ?? [] },
          }));
          setTranslationStatus("Summary translated. Translating extracted text...");
        } else if (event.type === "text_progress") {
          setTranslationProgress({ completed: event.completed, total: event.total });
          if (event.chunk) {
            setTranslations((current) => ({
              ...current,
              [key]: {
                ...current[key],
                language,
                summary: current[key]?.summary ?? null,
                extracted_text: null,
                extracted_chunks: [...(current[key]?.extracted_chunks ?? []), event.chunk],
              },
            }));
          }
        } else if (event.type === "complete") {
          setTranslations((current) => ({
            ...current,
            [key]: {
              ...current[key],
              language,
              summary: current[key]?.summary ?? null,
              extracted_text: event.extracted_text,
              extracted_chunks: [],
            },
          }));
          setTranslationStatus("");
          setTranslationProgress(null);
        }
      });
    } catch (e) {
      if (translationSessionRef.current === sessionId) {
        setTranslationError(e instanceof Error ? e.message : "Translation failed");
      }
    } finally {
      if (translationSessionRef.current === sessionId) {
        setTranslationLoadingKey((current) => current === key ? null : current);
        setTranslationStatus("");
        setTranslationProgress(null);
      }
    }
  }

  return <main className="app-shell"><header className="row"><h1>Welcome, {user.name}</h1><button onClick={() => void logout().catch((e) => setError(e instanceof Error ? e.message : "Could not log out"))}>Log out</button></header>
    <div className="workspace">
      <aside className="sidebar"><h2>Uploads</h2>{documents.length === 0 ? <p>No uploads yet.</p> : <nav aria-label="Uploaded media">{documents.map((document) => <button className={`document-link ${document.id === selectedId ? "selected" : ""}`} key={document.id} onClick={() => setSelectedId(document.id)}>{document.original_filename}</button>)}</nav>}</aside>
      <section className="content-panel">
        <section className="card"><h2>Extract text</h2><form ref={uploadFormRef} onSubmit={upload}><input name="file" type="file" accept=".pdf,image/png,image/jpeg,image/webp,image/bmp,image/tiff" required /><button disabled={busy}>{busy ? "Working..." : "Upload and extract"}</button></form></section>
        {error && <p className="error">{error}</p>}
        {selectedDocument ? <article className="card"><div className="row"><div><h2>{selectedDocument.original_filename}</h2><small>{new Date(selectedDocument.created_at).toLocaleString()}</small></div><button className="danger" disabled={busy} onClick={() => removeDocument(selectedDocument)}>Delete record</button></div>
          <nav className="language-tabs" role="tablist" aria-label="Report language">
            {LANGUAGES.map(({ code, label }) => <button
              key={code}
              id={`language-tab-${code}`}
              type="button"
              role="tab"
              aria-selected={activeLanguage === code}
              aria-controls="translated-report"
              className={activeLanguage === code ? "active" : ""}
              onClick={() => void changeLanguage(code)}
            >{label}</button>)}
          </nav>
          {translationKey !== null && translationLoadingKey === translationKey && <p role="status">{translationStatus || "Translating locally..."}{translationProgress && translationProgress.total > 0 ? ` (${translationProgress.completed}/${translationProgress.total} text chunks)` : ""}</p>}
          {translationError && <p className="error" role="alert">{translationError}</p>}
          <div id="translated-report" role="tabpanel" aria-labelledby={`language-tab-${activeLanguage}`}>
            <section aria-labelledby="summary-heading"><h3 id="summary-heading">Source-grounded report summary</h3>
              {activeLanguage === "en" ? (selectedDocument.summary ? selectedDocument.summary.split(/\n\s*\n/).filter(Boolean).map((paragraph, index) => <p key={index}>{paragraph}</p>) : <p>{selectedDocument.extracted_text ? "Summary unavailable for this existing report." : "No text was detected, so a summary could not be generated."}</p>) : activeTranslation?.summary ? activeTranslation.summary.split(/\n\s*\n/).filter(Boolean).map((paragraph, index) => <p key={index}>{paragraph}</p>) : <p>{translationError ? "Summary translation unavailable." : translationLoadingKey === translationKey ? "Translating summary..." : "Choose a language to translate this report."}</p>}
              <small>Informational only, not medical advice. Translations may not preserve every clinical nuance; verify important details against the original report.</small>
              <small>English-to-language translation runs on the MedLingua backend using its downloaded model; no translation API is used.</small>
            </section>
            <section aria-labelledby="extracted-text-heading"><h3 id="extracted-text-heading">Extracted text</h3>
              <pre>{activeLanguage === "en" ? (selectedDocument.extracted_text || "No text was detected.") : activeTranslation?.extracted_text || (activeTranslation?.extracted_chunks.length ? activeTranslation.extracted_chunks.join("\n\n") : translationError ? "Text translation unavailable. Switch to English to view the original." : translationLoadingKey === translationKey ? "Translating extracted text..." : "Choose a language to translate the extracted text.")}</pre>
            </section>
          </div>
        </article> : <section className="card"><p>Upload a PDF or image to view its extracted text here.</p></section>}
      </section>
    </div>
  </main>;
}
