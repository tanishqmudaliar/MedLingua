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

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 36 36" fill="none">
        <path d="M18 4.5v27M4.5 18h27M8.45 8.45l19.1 19.1m0-19.1-19.1 19.1" />
        <circle cx="18" cy="18" r="5.2" />
      </svg>
    </span>
  );
}

function FileGlyph({ image = false }: { image?: boolean }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      {image ? (
        <>
          <rect x="3.5" y="4" width="17" height="16" rx="2.5" />
          <circle cx="9" cy="10" r="1.6" />
          <path d="m5 17 4.4-4.3a1 1 0 0 1 1.4 0l2.2 2.1 1.7-1.6a1 1 0 0 1 1.4 0l2.9 2.8" />
        </>
      ) : (
        <>
          <path d="M6 3.75h7l5 5v11.5H6a2 2 0 0 1-2-2v-12a2.5 2.5 0 0 1 2-2.5Z" />
          <path d="M13 4v5h5M8 13h8m-8 3.5h8" />
        </>
      )}
    </svg>
  );
}

function UploadGlyph() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 15V4m0 0L8 8m4-4 4 4" />
      <path d="M5 14v4.25A1.75 1.75 0 0 0 6.75 20h10.5A1.75 1.75 0 0 0 19 18.25V14" />
    </svg>
  );
}

function Spinner() {
  return <span className="spinner" aria-hidden="true" />;
}

function initials(name: string): string {
  return name
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? "")
    .join("");
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(new Date(value));
}

export default function Home() {
  const [user, setUser] = useState<User | null>(null);
  const [checkingSession, setCheckingSession] = useState(true);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [selectedFilename, setSelectedFilename] = useState("");
  const [activeLanguage, setActiveLanguage] = useState<"en" | TranslationLanguage>("en");
  const [translations, setTranslations] = useState<Record<string, DocumentTranslation>>({});
  const [translationLoadingKey, setTranslationLoadingKey] = useState<string | null>(null);
  const [translationError, setTranslationError] = useState("");
  const [translationStatus, setTranslationStatus] = useState("");
  const [translationProgress, setTranslationProgress] = useState<{ completed: number; total: number } | null>(null);
  const uploadFormRef = useRef<HTMLFormElement>(null);
  const translationSessionRef = useRef(0);

  useEffect(() => {
    api.me()
      .then(setUser)
      .catch(() => undefined)
      .finally(() => setCheckingSession(false));
  }, []);

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
    event.preventDefault();
    setError("");
    setBusy(true);
    const data = Object.fromEntries(new FormData(event.currentTarget));
    try {
      const nextUser = mode === "login"
        ? await api.login({ username: String(data.username), password: String(data.password) })
        : await api.signup({
          username: String(data.username),
          name: String(data.name),
          email: String(data.email),
          password: String(data.password),
        });
      setUser(nextUser);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        setMode("signup");
        setError("Account not found. Sign up to create your account.");
      } else {
        setError(e instanceof Error ? e.message : "Unable to authenticate");
      }
    } finally {
      setBusy(false);
    }
  }

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    const file = (event.currentTarget.elements.namedItem("file") as HTMLInputElement).files?.[0];
    if (!file) return;
    setBusy(true);
    try {
      const document = await api.upload(file);
      setDocuments((current) => [document, ...current]);
      setSelectedId(document.id);
      uploadFormRef.current?.reset();
      setSelectedFilename("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function removeDocument(document: Document) {
    if (!window.confirm(`Delete "${document.original_filename}"? This cannot be undone.`)) return;
    setError("");
    setBusy(true);
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
    setDocuments([]);
    setSelectedId(null);
    setTranslations({});
    setActiveLanguage("en");
    setTranslationLoadingKey(null);
    setTranslationError("");
    setTranslationStatus("");
    setTranslationProgress(null);
  }

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
            [key]: {
              ...current[key],
              language,
              summary: event.text,
              extracted_text: current[key]?.extracted_text ?? null,
              extracted_chunks: current[key]?.extracted_chunks ?? [],
            },
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

  if (checkingSession) {
    return (
      <main className="session-check" aria-live="polite">
        <BrandMark />
        <Spinner />
        <span>Preparing your workspace</span>
      </main>
    );
  }

  if (!user) {
    return (
      <main className="auth-screen">
        <section className="auth-story" aria-label="About MedLingua">
          <a className="brand-lockup" href="/" aria-label="MedLingua home">
            <BrandMark />
            <span>medlingua</span>
          </a>
          <div className="auth-story-content">
            <span className="eyebrow eyebrow-light">CLARITY, IN EVERY REPORT</span>
            <h1>Understand your health information.</h1>
            <p>Medical documents, made easier to read. Extract, summarize, and translate your reports in one private workspace.</p>
            <div className="story-report" aria-hidden="true">
              <div className="story-report-top"><span /><span /><span /></div>
              <div className="story-report-title" />
              <div className="story-report-line story-report-line-short" />
              <div className="story-report-line" />
              <div className="story-report-line story-report-line-mid" />
              <div className="story-report-highlight"><span /><div><i /><i /><i /></div></div>
              <div className="story-report-line" />
              <div className="story-report-line story-report-line-short" />
            </div>
          </div>
          <div className="auth-story-footer"><span>YOUR REPORTS. YOUR DEVICE. YOUR PACE.</span><span>01 — 04</span></div>
        </section>

        <section className="auth-panel">
          <div className="auth-panel-inner">
            <div className="auth-mobile-brand">
              <BrandMark />
              <span>medlingua</span>
            </div>
            <div className="auth-form-heading">
              <span className="eyebrow">{mode === "login" ? "WELCOME BACK" : "GET STARTED"}</span>
              <h2>{mode === "login" ? "Sign in to your workspace" : "Create your account"}</h2>
              <p>{mode === "login" ? "Pick up where you left off." : "A clearer way to read your medical reports."}</p>
            </div>
            <form className="auth-form" onSubmit={submitAuth}>
              <label htmlFor="username">Username</label>
              <input
                id="username"
                name="username"
                placeholder="Enter your username"
                autoComplete="username"
                required
                minLength={3}
              />
              {mode === "signup" && (
                <>
                  <label htmlFor="name">Full name</label>
                  <input id="name" name="name" placeholder="How should we address you?" autoComplete="name" required />
                  <label htmlFor="email">Email address</label>
                  <input id="email" name="email" type="email" placeholder="you@example.com" autoComplete="email" required />
                </>
              )}
              <label htmlFor="password">Password</label>
              <input
                id="password"
                name="password"
                type="password"
                placeholder={mode === "signup" ? "At least 8 characters" : "Enter your password"}
                autoComplete={mode === "signup" ? "new-password" : "current-password"}
                required
                minLength={mode === "signup" ? 8 : undefined}
              />
              {error && <p className="form-alert" role="alert">{error}</p>}
              <button className="button button-primary auth-submit" disabled={busy}>
                {busy ? <><Spinner /> {mode === "login" ? "Signing in..." : "Creating account..."}</> : mode === "login" ? "Sign in" : "Create account"}
                {!busy && <span aria-hidden="true">↗</span>}
              </button>
            </form>
            <p className="auth-switch">
              {mode === "login" ? "New to MedLingua?" : "Already have an account?"}
              <button
                type="button"
                onClick={() => { setMode(mode === "login" ? "signup" : "login"); setError(""); }}
              >
                {mode === "login" ? "Create an account" : "Sign in"}
              </button>
            </p>
            <p className="auth-disclaimer">For informational purposes only. Always verify details with your healthcare professional.</p>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <aside className="app-sidebar">
        <a className="brand-lockup sidebar-brand" href="/" aria-label="MedLingua home">
          <BrandMark />
          <span>medlingua</span>
        </a>

        <div className="sidebar-section-label">
          <span>YOUR WORKSPACE</span>
          <span className="sidebar-live-mark" title="Workspace ready" />
        </div>
        <div className="sidebar-current">
          <span className="sidebar-current-icon"><FileGlyph /></span>
          <span>Report library</span>
          <span className="sidebar-count">{documents.length}</span>
        </div>

        <div className="library-heading">
          <span>RECENT REPORTS</span>
          <span>{String(documents.length).padStart(2, "0")}</span>
        </div>
        <nav className="document-nav" aria-label="Uploaded reports">
          {documents.length === 0 ? (
            <p className="sidebar-empty">Your reports will appear here after upload.</p>
          ) : documents.map((document) => (
            <button
              className={`document-link ${document.id === selectedId ? "selected" : ""}`}
              key={document.id}
              onClick={() => setSelectedId(document.id)}
              aria-current={document.id === selectedId ? "page" : undefined}
              title={document.original_filename}
            >
              <span className="document-link-icon"><FileGlyph image={document.media_type.startsWith("image/")} /></span>
              <span className="document-link-copy">
                <span className="document-link-name">{document.original_filename}</span>
                <span className="document-link-date">{formatDate(document.created_at)}</span>
              </span>
              {document.id === selectedId && <span className="document-selected-mark" aria-hidden="true" />}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="account-card">
            <span className="avatar">{initials(user.name)}</span>
            <span className="account-copy"><strong>{user.name}</strong><small>@{user.username}</small></span>
            <button
              className="logout-button"
              type="button"
              aria-label="Log out"
              title="Log out"
              onClick={() => void logout().catch((e) => setError(e instanceof Error ? e.message : "Could not log out"))}
            >
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M10 5H6.75A1.75 1.75 0 0 0 5 6.75v10.5A1.75 1.75 0 0 0 6.75 19H10m4-3 4-4-4-4m4 4H9" />
              </svg>
            </button>
          </div>
          <p className="sidebar-footnote">Medical clarity, thoughtfully delivered.</p>
        </div>
      </aside>

      <section className="main-area">
        <header className="page-header">
          <div>
            <div className="breadcrumb"><span>WORKSPACE</span><span className="breadcrumb-divider">/</span><span>REPORTS</span></div>
            <h1>Your reports <span className="heading-count">{documents.length}</span></h1>
            <p className="page-subtitle">A clearer view of the information in your medical documents.</p>
          </div>
          <div className="header-stamp"><span className="stamp-dot" /> PERSONAL LIBRARY</div>
        </header>

        <div className="main-content">
          <section className="upload-card" aria-labelledby="upload-title">
            <div className="upload-card-copy">
              <span className="upload-icon"><UploadGlyph /></span>
              <div>
                <h2 id="upload-title">Add a report</h2>
                <p>Choose a PDF or image to extract and understand its contents.</p>
              </div>
            </div>
            <form className="upload-form" ref={uploadFormRef} onSubmit={upload}>
              <label className="file-picker">
                <FileGlyph />
                <span className="file-picker-name">{selectedFilename || "Choose file"}</span>
                <input
                  name="file"
                  type="file"
                  accept=".pdf,image/png,image/jpeg,image/webp,image/bmp,image/tiff"
                  required
                  aria-label="Choose a medical report file"
                  onChange={(event) => setSelectedFilename(event.currentTarget.files?.[0]?.name ?? "")}
                />
              </label>
              <button className="button button-primary upload-submit" disabled={busy}>
                {busy ? <><Spinner /> Processing...</> : "Upload report"}
                {!busy && <span aria-hidden="true">↗</span>}
              </button>
            </form>
          </section>

          {error && <p className="global-alert" role="alert">{error}</p>}

          {selectedDocument ? (
            <article className="report-view">
              <header className="report-header">
                <div className="report-heading">
                  <span className="eyebrow">REPORT OVERVIEW</span>
                  <div className="report-title-row">
                    <span className="report-file-icon"><FileGlyph image={selectedDocument.media_type.startsWith("image/")} /></span>
                    <h2 title={selectedDocument.original_filename}>{selectedDocument.original_filename}</h2>
                  </div>
                  <p className="report-date">Added {formatDate(selectedDocument.created_at)}</p>
                </div>
                <button
                  className="button button-danger-ghost"
                  type="button"
                  disabled={busy}
                  onClick={() => void removeDocument(selectedDocument)}
                >
                  <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path d="M4.5 7h15m-13.5 0 .8 12.1a1.5 1.5 0 0 0 1.5 1.4h7.4a1.5 1.5 0 0 0 1.5-1.4L18 7M9.5 10.5v6m5-6v6M9 7V4.75A.75.75 0 0 1 9.75 4h4.5a.75.75 0 0 1 .75.75V7" />
                  </svg>
                  Delete report
                </button>
              </header>

              <nav className="language-tabs" role="tablist" aria-label="Report language">
                {LANGUAGES.map(({ code, label }) => (
                  <button
                    key={code}
                    id={`language-tab-${code}`}
                    type="button"
                    role="tab"
                    aria-selected={activeLanguage === code}
                    aria-controls="translated-report"
                    className={activeLanguage === code ? "active" : ""}
                    onClick={() => void changeLanguage(code)}
                  >{label}</button>
                ))}
                <span className="language-note">Translations are generated locally</span>
              </nav>

              {translationKey !== null && translationLoadingKey === translationKey && (
                <div className="translation-status" role="status" aria-live="polite">
                  <Spinner />
                  <span>{translationStatus || "Translating locally..."}</span>
                  {translationProgress && translationProgress.total > 0 && (
                    <span className="progress-count">{translationProgress.completed}/{translationProgress.total} sections</span>
                  )}
                </div>
              )}
              {translationError && <p className="global-alert translation-alert" role="alert">{translationError}</p>}

              <div id="translated-report" className="report-content" role="tabpanel" aria-labelledby={`language-tab-${activeLanguage}`}>
                <section className="summary-card" aria-labelledby="summary-heading">
                  <div className="section-kicker">
                    <span className="section-index">01</span>
                    <span>THE ESSENTIALS</span>
                    <span className="section-rule" />
                    <span className="section-label">SUMMARY</span>
                  </div>
                  <div className="summary-body">
                    <h3 id="summary-heading">The report, made clearer.</h3>
                    <div className="summary-copy">
                      {activeLanguage === "en" ? (
                        selectedDocument.summary
                          ? selectedDocument.summary.split(/\n\s*\n/).filter(Boolean).map((paragraph, index) => <p key={index}>{paragraph}</p>)
                          : <p className="muted-copy">{selectedDocument.extracted_text ? "A summary is unavailable for this report." : "No text was detected, so a summary could not be generated."}</p>
                      ) : activeTranslation?.summary ? (
                        activeTranslation.summary.split(/\n\s*\n/).filter(Boolean).map((paragraph, index) => <p key={index}>{paragraph}</p>)
                      ) : (
                        <p className="muted-copy">
                          {translationError
                            ? "Summary translation unavailable."
                            : translationLoadingKey === translationKey
                              ? "Translating the summary..."
                              : "Choose a language to translate this report."}
                        </p>
                      )}
                    </div>
                    <div className="summary-disclaimer">
                      <span className="disclaimer-mark" aria-hidden="true">i</span>
                      <p>For information only, not medical advice. Verify important details with the original report and a healthcare professional.</p>
                    </div>
                  </div>
                </section>

                <section className="extracted-card" aria-labelledby="extracted-text-heading">
                  <div className="extracted-header">
                    <div>
                      <div className="section-kicker">
                        <span className="section-index">02</span>
                        <span>THE SOURCE</span>
                        <span className="section-rule" />
                        <span className="section-label">DOCUMENT TEXT</span>
                      </div>
                      <h3 id="extracted-text-heading">Extracted text</h3>
                    </div>
                    <span className="source-language">{activeLanguage === "en" ? "ORIGINAL · EN" : `TRANSLATED · ${activeLanguage.toUpperCase()}`}</span>
                  </div>
                  <pre>{activeLanguage === "en"
                    ? (selectedDocument.extracted_text || "No text was detected.")
                    : activeTranslation?.extracted_text || (activeTranslation?.extracted_chunks.length
                      ? activeTranslation.extracted_chunks.join("\n\n")
                      : translationError
                        ? "Text translation unavailable. Switch to English to view the original."
                        : translationLoadingKey === translationKey
                          ? "Translating extracted text..."
                          : "Choose a language to translate the extracted text.")}</pre>
                </section>
              </div>
              <footer className="report-footer">
                <span><BrandMark /> MedLingua</span>
                <span>Always compare summaries and translations with the original document.</span>
              </footer>
            </article>
          ) : (
            <section className="empty-workspace" aria-labelledby="empty-title">
              <div className="empty-illustration" aria-hidden="true">
                <div className="empty-paper empty-paper-back" />
                <div className="empty-paper empty-paper-front">
                  <span /><span /><span /><i />
                </div>
                <span className="empty-spark empty-spark-one" />
                <span className="empty-spark empty-spark-two" />
              </div>
              <span className="eyebrow">A CLEARER PICTURE STARTS HERE</span>
              <h2 id="empty-title">{documents.length ? "Select a report to begin." : "Your reports, understood."}</h2>
              <p>{documents.length
                ? "Choose a report from your library to review its summary and extracted text."
                : "Upload a medical report to extract its text, explore a plain-language summary, and translate it when you need to."}</p>
              <span className="empty-caption"><span className="stamp-dot" /> PRIVATE BY DESIGN · LOCAL PROCESSING</span>
            </section>
          )}
          <footer className="workspace-footer">
            <span>MEDLINGUA <span className="footer-divider">/</span> PERSONAL HEALTH DOCUMENTS</span>
            <span>INFORMATIONAL PURPOSES ONLY</span>
          </footer>
        </div>
      </section>
    </main>
  );
}
