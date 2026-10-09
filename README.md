# MedLingua

MedLingua turns medical PDFs and images into extracted text, a source-grounded plain-language summary, and locally generated Hindi, Marathi, or Tamil translations.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Next.js](https://img.shields.io/badge/Frontend-Next.js%2015-black)
![Database](https://img.shields.io/badge/Database-PostgreSQL-4169E1)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage Guide](#usage-guide)
- [AI and Translation Pipeline](#ai-and-translation-pipeline)
- [API Reference](#api-reference)
- [Database Schema and Migrations](#database-schema-and-migrations)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Deployment Considerations](#deployment-considerations)
- [Privacy and Security](#privacy-and-security)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

MedLingua is a local-first web application for extracting, reviewing, summarizing, and translating text from medical documents. It combines a Next.js interface with a FastAPI backend and PostgreSQL storage. OCR, extractive summarization, and machine translation run in the backend; the translation model uses a compatible NVIDIA GPU when available and otherwise runs on CPU.

The project is intended to make reports easier to read, not to interpret them in place of a clinician. The original extracted text remains available alongside the summary and translations so users can compare results with the source.

## Features

### Accounts and document library

- Sign up and log in with a username and password.
- Documents are associated with the authenticated account.
- Browse, select, and delete uploaded records.
- Authentication uses a signed, HttpOnly cookie.

### Extraction and summaries

- Upload PDFs and PNG, JPEG, WEBP, BMP, or TIFF images.
- Extract embedded PDF text with PyMuPDF and use OCR for scanned PDF pages and images.
- Apply image preprocessing and basic column-aware reading order to improve OCR output.
- Generate an English, multi-paragraph extractive summary that prioritizes report sections such as symptoms and timeline, medical history, findings, assessment, and plan.
- Keep the extracted text and summary on the document record for later viewing.

### Translations

- Switch between English, Hindi, Marathi, and Tamil from the report language tabs.
- Translate both the summary and extracted text with the locally downloaded M2M100 model.
- Stream the translated summary and extracted-text chunks to the browser with progress updates.
- Persist translated summaries and chunks per document and language, so unfinished work can resume and completed translations do not need to be generated again.
- Automatically select CUDA when PyTorch can access a CUDA-compatible NVIDIA GPU; otherwise, use CPU.

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend | Next.js 15, React 19, TypeScript |
| Backend API | Python 3.11, FastAPI, Uvicorn |
| Persistence | PostgreSQL 16, SQLAlchemy 2, asyncpg, Alembic |
| PDF and image processing | PyMuPDF, OpenCV, PaddleOCR |
| Summarization | Local Python NLP and extractive sentence selection |
| Translation | Hugging Face Transformers, M2M100, PyTorch 2.10 CUDA 12.8 build, SentencePiece |
| Tests | pytest, TypeScript compiler |

## Getting Started

### One-command startup on Windows

From the repository root, double-click `run_medlingua.bat` or run it from Command Prompt. It creates missing local environment files and the backend virtual environment, installs backend and frontend dependencies, starts PostgreSQL with Docker Compose, applies Alembic migrations, checks/downloads the local translation model, and opens the backend and frontend development servers in separate windows.

Install and start Docker Desktop, and install Python 3.11, Node.js, and npm before running the script. The first run can take a while: it downloads the CUDA-enabled PyTorch wheel and the translation model. Keep the two server windows open while using the app. The script does not overwrite existing `.env` or `.env.local` files.

### Prerequisites

- Windows 10/11 or another supported Python and Node.js environment.
- Python 3.11 and pip.
- Node.js and npm.
- Docker Desktop with Docker Compose, or a separately installed PostgreSQL 16 server.
- Git.
- For GPU translation: an NVIDIA GPU, a compatible NVIDIA driver, and a CUDA-enabled PyTorch installation. This repository pins the CUDA 12.8 PyTorch 2.10 wheel. The GTX 1650 has 4 GB of VRAM, which can limit the size of inputs that fit on the GPU.
- A TeX distribution such as MiKTeX or TeX Live only if you want to compile the included `demo/demo.tex`.

### Installation

The following instructions use Windows Command Prompt (CMD). Run commands from the indicated directory. Do not paste the Markdown fence lines.

#### 1. Start PostgreSQL

From the repository root:

```cmd
docker compose up -d postgres
```

The Compose service creates a local development database named `medlingua` on port `5432`. Its sample credentials are for local development only; do not expose them on a public or production database.

#### 2. Configure and install the backend

From the repository root:

```cmd
cd backend
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `backend\.env` before starting the API:

- Set `DATABASE_URL` to the PostgreSQL connection string. For the repository's local Compose database, use `postgresql+asyncpg://<user>:<password>@localhost:5432/medlingua`, with the local development values configured in `docker-compose.yml`.
- Replace `JWT_SECRET_KEY` with a private random value. For example, while the backend virtual environment is active:

  ```cmd
  python -c "import secrets; print(secrets.token_urlsafe(48))"
  ```

Run the printed command again only if you need a new secret; do not commit the resulting secret to Git.

#### 3. Apply database migrations

From the `backend` directory with the virtual environment active:

```cmd
alembic upgrade head
```

#### 4. Download the translation model

This is a one-time download for a new Hugging Face cache. It requires an internet connection and approximately 1.8 GB of disk space:

```cmd
python -m app.download_translation_model
```

The English OCR model files are initialized by PaddleOCR when OCR is first used.

#### 5. Start the backend

Keep this terminal open:

```cmd
python -m uvicorn app.main:app --reload --port 8000
```

The API is available at `http://localhost:8000`. Interactive API documentation is available at `http://localhost:8000/docs`.

#### 6. Configure and start the frontend

Open a second terminal at the repository root:

```cmd
cd frontend
copy .env.local.example .env.local
npm ci
npm run dev
```

Open `http://localhost:3000`. Keep the frontend terminal open while using the development server.

### Optional: compile the sample LaTeX report

Install MiKTeX or TeX Live, then run from the repository root:

```cmd
cd demo
pdflatex demo.tex
```

This produces `demo.pdf` in the `demo` folder, which can then be uploaded through MedLingua.

## Usage Guide

1. Open `http://localhost:3000`.
2. Create an account or log in. Usernames must be 3–32 letters, digits, periods, underscores, or hyphens. Signup passwords must be 8–128 characters.
3. Select a PDF or supported image and choose **Upload and extract**. Uploads are limited by `MAX_UPLOAD_SIZE_MB`; PDFs are limited by `MAX_PDF_PAGES`.
4. Select an uploaded report from the left-side list. Review its extracted text and English summary.
5. Choose **हिन्दी**, **मराठी**, or **தமிழ்** to request a translation. The translated summary is shown first; extracted text follows as chunks finish. If a translation is interrupted, the saved chunks can be reused on the next request.
6. Return to English to compare the generated text with the source. Use **Delete record** to remove a report and its stored upload.
7. Log out when finished. The browser session uses the authentication cookie; document data remains in the database until deleted.

Translation quality and speed vary with the report length, device, and model. Reopening a language already translated for a report should reuse its saved result.

## AI and Translation Pipeline

### Upload, extraction, and summary

1. The API validates the media type, upload size, and PDF page limit.
2. The uploaded file is written to the configured upload directory.
3. For PDFs, PyMuPDF extracts embedded text page-by-page. Pages without embedded text are rendered to images and sent through OCR. Uploaded images are preprocessed with OpenCV before OCR.
4. Extracted text is stored on the document record. For two-column pages, extraction attempts to read the left column followed by the right column.
5. The summary code cleans and ranks source sentences, selects useful content from recognized report sections, and adds a limited set of plain-language term explanations. It is extractive; it does not call a hosted generative AI service.
6. The extracted text and summary are returned to the browser and persisted with the report.

### Translation and cache

- The backend uses `facebook/m2m100_418M` from the local Hugging Face cache. Model download is a separate setup step; inference does not send report text to Hugging Face or a translation API.
- The model is loaded lazily when translation is first requested. If PyTorch reports CUDA as available at model load, the model and inference tensors are placed on CUDA; otherwise, they stay on CPU.
- The source report is divided into smaller text chunks. The endpoint streams newline-delimited JSON (NDJSON) events for model status, summary, chunk progress, completion, or an error.
- Translations are cached in the document's PostgreSQL JSONB `translations` field by language (`hi`, `mr`, or `ta`). The translated summary and each completed extracted-text chunk are written as they finish; a completed full translation is stored as well.
- Cached translations are tied to the current document text. If source text is changed outside the normal application flow, previously cached translations should be considered stale.

## API Reference

The FastAPI server defaults to `http://localhost:8000`. Authenticated routes use the `access_token` HttpOnly cookie set by signup or login. JSON request bodies use `Content-Type: application/json`, except uploads, which use `multipart/form-data`.

| Method | Endpoint | Authentication | Purpose |
|---|---|---|---|
| `GET` | `/health` | No | Basic API liveness check |
| `POST` | `/auth/signup` | No | Create an account and set the login cookie |
| `POST` | `/auth/login` | No | Authenticate and set the login cookie |
| `POST` | `/auth/logout` | No | Clear the login cookie |
| `GET` | `/auth/me` | Yes | Return the current account |
| `POST` | `/documents` | Yes | Upload a PDF or image, extract text, and create its summary |
| `GET` | `/documents` | Yes | List the current account's documents |
| `GET` | `/documents/{document_id}` | Yes | Read one of the current account's documents |
| `POST` | `/documents/{document_id}/translate` | Yes | Stream/cache a translation |
| `DELETE` | `/documents/{document_id}` | Yes | Delete the document and its uploaded file |

### Request examples

Signup:

```json
{
  "username": "reader_01",
  "name": "Report Reader",
  "email": "reader@example.com",
  "password": "use-a-private-password"
}
```

Login:

```json
{
  "username": "reader_01",
  "password": "use-a-private-password"
}
```

Translation (the only accepted target codes are `hi`, `mr`, and `ta`):

```json
{
  "language": "hi"
}
```

Upload a file using the multipart field name `file`.

### Document response

Document responses include the document ID, original filename, media type, extracted text, nullable English summary, creation timestamp, and a `translations` object. Translation entries can contain a nullable translated summary, nullable completed extracted text, and `extracted_chunks` saved so an interrupted job can resume. Translation responses themselves are streamed as one JSON object per line; event types include `status`, `summary`, `text_progress`, `complete`, and `error`.

### Common HTTP errors

| Status | Meaning |
|---|---|
| `401` | Missing or invalid login cookie |
| `404` | Requested document or login account was not found |
| `409` | Signup username or email is already registered |
| `413` | Upload exceeds the configured size limit |
| `415` | Unsupported upload media type |
| `422` | Invalid request data or invalid/over-limit PDF |
| `503` | OCR runtime dependency is unavailable |
| `500` | Unexpected translation processing failure; check backend logs |

When the translation model is unavailable, the streaming endpoint sends an `error` NDJSON event whose `status` field is `503`; the HTTP response may already have started with status `200`.

## Database Schema and Migrations

MedLingua uses PostgreSQL. Alembic owns the schema history; apply all pending migrations from the `backend` directory:

```cmd
alembic upgrade head
```

Current tables:

| Table | Purpose |
|---|---|
| `users` | Account ID, username, display name, email, password hash, and creation timestamp |
| `documents` | Owner, filename, media type, upload path, extracted text, summary, translation cache, and creation timestamp |
| `alembic_version` | Current migration revision |

Migration history:

| Revision | Change |
|---|---|
| `0001_initial` | Create users and documents |
| `0002_document_summary` | Add the nullable English summary |
| `0003_document_translations` | Add a non-null PostgreSQL JSONB translation cache |

The document's `owner_id` references `users.id`. API document lookups are scoped to the authenticated owner. The database and upload directory are separate persistence stores; protect and back up both if retaining reports matters. Do not delete the database volume or upload files unless you intend to permanently remove the stored data.

## Project Structure

```text
MedLingua/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI routes and application wiring
│   │   ├── config.py                # Environment-backed settings
│   │   ├── db.py                    # SQLAlchemy engine and sessions
│   │   ├── models.py                # User and document ORM models
│   │   ├── schemas.py               # API request and response schemas
│   │   ├── security.py              # Password hashing and JWT helpers
│   │   ├── dependencies.py          # Authenticated-user dependency
│   │   ├── ocr.py                   # PDF/image extraction and OCR
│   │   ├── summary.py               # Source-grounded extractive summaries
│   │   ├── translation.py           # Local M2M100 model and chunking
│   │   └── download_translation_model.py
│   ├── migrations/
│   │   ├── env.py                   # Alembic database configuration
│   │   └── versions/                # Alembic schema revisions
│   ├── tests/                       # Summary and translation tests
│   ├── uploads/                     # Default local upload storage
│   ├── alembic.ini
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app/
│   │   ├── page.tsx                 # Login, uploads, language tabs, report UI
│   │   ├── layout.tsx               # App metadata and root layout
│   │   └── globals.css              # Global UI styles
│   ├── lib/api.ts                   # API types, requests, and stream parser
│   ├── package.json
│   └── .env.local.example
├── demo/
│   └── demo.tex                     # Sample report for local PDF generation
├── docker-compose.yml               # Local PostgreSQL service
└── README.md
```

## Configuration

Backend settings are loaded from `backend/.env` (when running commands from `backend`) or from the process environment. Copy `backend/.env.example` to start. Frontend settings are loaded from `frontend/.env.local`.

| Variable | Component | Required | Default / example | Description |
|---|---|---:|---|---|
| `DATABASE_URL` | Backend | Yes | `postgresql+asyncpg://<user>:<password>@localhost:5432/medlingua` | Async SQLAlchemy PostgreSQL connection string |
| `JWT_SECRET_KEY` | Backend | Yes | Replace with a private random value | Secret used to sign authentication tokens |
| `JWT_EXPIRE_MINUTES` | Backend | No | `1440` | Authentication token lifetime in minutes |
| `UPLOAD_DIR` | Backend | No | `./uploads` | Directory where uploaded files are stored; relative paths are resolved from the backend process working directory |
| `MAX_UPLOAD_SIZE_MB` | Backend | No | `20` | Maximum accepted upload size in MiB |
| `MAX_PDF_PAGES` | Backend | No | `20` | Maximum number of pages accepted in a PDF |
| `FRONTEND_ORIGIN` | Backend | No | `http://localhost:3000` | Allowed frontend origin for CORS |
| `NEXT_PUBLIC_API_URL` | Frontend | No | `http://localhost:8000` | Base URL used by browser API requests |

When changing `FRONTEND_ORIGIN` or `NEXT_PUBLIC_API_URL`, keep them consistent with the actual frontend and backend addresses. Restart the relevant development server after editing environment files.

## Deployment Considerations

The repository currently documents and tests local development; it does not include a production deployment configuration. Before exposing an instance to users:

- Use HTTPS, a strong private `JWT_SECRET_KEY`, production-grade PostgreSQL credentials, and a persistent database with tested backups.
- Configure a persistent, access-restricted upload directory and include it in your backup and retention plan. Database backups alone do not contain the original uploaded files.
- Set `FRONTEND_ORIGIN` to the exact public frontend origin and set `NEXT_PUBLIC_API_URL` to the browser-reachable API base URL. Because this value is compiled into the Next.js frontend, set it before building the production frontend.
- Run Alembic migrations as a controlled deployment step. Run Uvicorn without `--reload` behind an HTTPS reverse proxy and run the frontend using a production Next.js server.
- Review cookie security settings before deployment. The current authentication cookie is HttpOnly and SameSite=Lax but is not configured with the `Secure` flag in the application code.
- Review upload limits, privacy and retention obligations, monitoring, access controls, and the handling of sensitive medical records before production use.

## Privacy and Security

- Passwords are stored as salted PBKDF2-SHA256 hashes; the raw password is not stored in the user table.
- Authentication uses an HS256-signed JWT stored in an HttpOnly cookie. Keep `JWT_SECRET_KEY` private and use HTTPS and production-grade cookie, secret, and database settings before any deployment.
- Document routes check ownership against the logged-in account. Uploaded documents and their extracted content are persisted locally in PostgreSQL and the configured upload directory.
- OCR and summarization run in the backend. Translation inference runs using locally cached model files; the model downloader needs internet access, but report text is not sent to a hosted translation API by the inference flow.
- CORS is configured for one frontend origin through `FRONTEND_ORIGIN`.
- The Compose PostgreSQL username/password are development defaults, not production credentials. Do not expose the development database port to untrusted networks.
- Medical reports can contain highly sensitive personal information. Restrict access to the host, database, backups, and upload directory; avoid using real patient information in demos or logs.
- Summaries and translations are informational aids, not medical advice, diagnosis, or treatment recommendations. Verify all important details against the original report and consult a qualified healthcare professional.

## Troubleshooting

| Problem | Likely cause | Resolution |
|---|---|---|
| Frontend says it cannot connect to the backend | API server is stopped or `NEXT_PUBLIC_API_URL` is wrong | Start Uvicorn on port 8000 and verify the frontend API URL. Check `http://localhost:8000/health`. |
| Database connection fails at startup or during a request | PostgreSQL is stopped, `DATABASE_URL` is incorrect, or migrations have not been applied | Start the Compose service, correct `DATABASE_URL`, then run `alembic upgrade head` from `backend`. |
| `alembic current head` reports an unrecognized argument | `current` and `head` are separate Alembic concepts | Use `alembic current` to inspect the applied revision and `alembic upgrade head` to apply all migrations. |
| Translation reports that the local model is unavailable | Dependencies or M2M100 model files are missing from the active Python environment/cache | Install `backend/requirements.txt` in the backend virtual environment and run `python -m app.download_translation_model` from `backend`. |
| Translation is still slow | Long source text, first model load, CPU inference, or constrained GPU resources | Allow the first model load to finish; verify PyTorch CUDA availability with `python -c "import torch; print(torch.cuda.is_available())"`. Cached chunks are reused on later requests. |
| CUDA out-of-memory error | The model or current inference input exceeds available GPU memory | Close other GPU workloads and retry. The current GTX 1650 has 4 GB VRAM; the application does not guarantee automatic CPU fallback after a CUDA out-of-memory error. |
| `alembic` or `uvicorn` command is not found on Windows | The backend virtual environment is not active or executable launchers are blocked | Activate `.venv\Scripts\activate` and use `python -m uvicorn app.main:app --reload --port 8000`; use `python -m pip` for package commands. |
| No text is extracted from a PDF | The PDF is image-only, protected, poor quality, or OCR cannot recognize its content | Confirm the page is within `MAX_PDF_PAGES`, upload a clearer PDF/image, and compare OCR output to the source. |
| Upload is rejected | Unsupported media type, file over size limit, or PDF over page limit | Use PDF/PNG/JPEG/WEBP/BMP/TIFF and check `MAX_UPLOAD_SIZE_MB` and `MAX_PDF_PAGES`. |

## Known Limitations

- OCR may omit or misread small, low-resolution, handwritten, rotated, or visually complex content. The column-order heuristic is not a full document-layout engine.
- Summary generation is extractive and section-aware, but its coverage, terminology explanations, and ranking are limited. It is not clinically validated and may miss details or inherit extraction errors.
- M2M100 translations can mistranslate medical terms, abbreviations, medication names, negation, and measurements. A translated output must not replace the source or a professional translation.
- The first translation request loads the model. GPU use requires a compatible NVIDIA driver and CUDA-enabled PyTorch; GPU memory is limited, and CPU inference is slower. Inference can still take minutes for long reports.
- The app processes one uploaded PDF/image at a time and supports a limited set of English-oriented OCR inputs.
- The README documents local development. No production deployment recipe, cloud backup system, user account recovery, or clinical compliance certification is included.
- Existing records created before summary or translation migrations may not have generated summaries or translation entries. Existing English extracted text remains available.

## Testing

From a terminal in `backend` with its virtual environment active:

```cmd
python -m pytest tests -q
```

From `frontend`:

```cmd
npm run lint
npm run build
```

The backend tests cover the extractive summary behavior and translation language selection, chunking, caching behavior, and CUDA-versus-CPU device selection.

## Contributing

1. Fork the repository and create a focused branch from the current default branch.
2. Make a focused change that preserves document ownership checks and keeps sensitive report text local.
3. Add or update tests for behavior changes.
4. Run the backend tests and frontend lint/build commands listed above.
5. Commit your changes with a clear message, push the branch to your fork, and open a pull request describing the change, validation performed, and any migration or setup implications.

Please do not include real patient reports, credentials, model cache files, virtual environments, or generated upload data in commits.

## License

No license file or license grant is currently included in this repository.

---

Made with ❤️ by [Tanishq Mudaliar](https://github.com/tanishqmudaliar)

Understand the report more clearly. Always verify against the original.
