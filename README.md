# MedLingua

MedLingua is a private, local-first medical natural language processing platform that transforms complex clinical reports (PDFs and images) into source-grounded plain-language summaries, vernacular translations (**Hindi**, **Marathi**, and **Tamil**), and conversational document QA on consumer hardware.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Next.js](https://img.shields.io/badge/Frontend-Next.js%2015-black)
![Database](https://img.shields.io/badge/Database-PostgreSQL%2016-4169E1)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10%20CUDA%2012.8-EE4C2C)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Llama%203.2%203B-black)
![IndicTrans2](https://img.shields.io/badge/NMT-AI4Bharat%20IndicTrans2-orange)
![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20GTX%201650%20(4GB)-76B900)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [System Overview & Walkthrough](#system-overview--walkthrough)
- [Verified Core Features](#verified-core-features)
- [Architecture & Data Flow](#architecture--data-flow)
- [Hardware Optimization & Memory Architecture](#hardware-optimization--memory-architecture)
- [NLP & Machine Learning Deep Dive](#nlp--machine-learning-deep-dive)
  - [1. Translation Architecture: AI4Bharat IndicTrans2 & Meta M2M100 Fallback](#1-translation-architecture-ai4bharat-indictrans2--meta-m2m100-fallback)
  - [2. Transformers 5.x Deep Compatibility Resolutions & Shims](#2-transformers-5x-deep-compatibility-resolutions--shims)
  - [3. SafeTensors vs. Legacy Pickle Checkpoints](#3-safetensors-vs-legacy-pickle-checkpoints)
  - [4. Text Processing & Boundary-Preserving Subword Chunking](#4-text-processing--boundary-preserving-subword-chunking)
  - [5. Source-Grounded Clinical Summarization: Mathematical & Algorithmic Formulation](#5-source-grounded-clinical-summarization-mathematical--algorithmic-formulation)
  - [6. Plain-Language Lexical Expansion Engine (42+ Medical Acronyms)](#6-plain-language-lexical-expansion-engine-42-medical-acronyms)
  - [7. Extractive vs. Abstractive Summarization & Clinical Safety Guardrails](#7-extractive-vs-abstractive-summarization--clinical-safety-guardrails)
  - [8. Empirical NLP Evaluation Framework](#8-empirical-nlp-evaluation-framework)
- [Sequential Translation Pipeline & Queue Architecture](#sequential-translation-pipeline--queue-architecture)
  - [1. Deterministic Multi-Language Pipeline Order](#1-deterministic-multi-language-pipeline-order)
  - [2. Cross-Tab Real-Time Queue Status Transparency](#2-cross-tab-real-time-queue-status-transparency)
  - [3. Click-Order FIFO Re-Translation Queue](#3-click-order-fifo-re-translation-queue)
  - [4. Persistent PostgreSQL Queue & Serialized Worker](#4-persistent-postgresql-queue--serialized-worker)
- [Document-Grounded Clinical Chatbot](#document-grounded-clinical-chatbot)
- [Real-Time Interactive Streaming Feeds](#real-time-interactive-streaming-feeds)
- [Complete Technology Stack & Dependency Audit](#complete-technology-stack--dependency-audit)
- [Automated Toolchain, Diagnostics & Orchestration](#automated-toolchain-diagnostics--orchestration)
- [Getting Started & Local Development](#getting-started--local-development)
- [Production Deployment Guide (Zero-Cost Cloud Architecture)](#production-deployment-guide-zero-cost-cloud-architecture)
- [API Reference](#api-reference)
- [Database Schema & Migrations](#database-schema--migrations)
- [Project Structure & Repository Map](#project-structure--repository-map)
- [Configuration & Environment Variables](#configuration--environment-variables)
- [Privacy, Security & Medical Disclaimer](#privacy-security--medical-disclaimer)
- [Testing Strategy & Test Suite Results](#testing-strategy--test-suite-results)
- [Troubleshooting & Compatibility Guide](#troubleshooting--compatibility-guide)
- [Deprecations, Replacements & Architecture Evolution](#deprecations-replacements--architecture-evolution)
- [Academic Viva Voce Defense, Demonstration & NLP Glossary](#academic-viva-voce-defense-demonstration--nlp-glossary)
  - [1. Two-Minute Elevator Pitch](#1-two-minute-elevator-pitch)
  - [2. Comprehensive NLP Glossary](#2-comprehensive-nlp-glossary)
  - [3. Top 25 Viva Voce Questions & Defensible Answers](#3-top-25-viva-voce-questions--defensible-answers)
  - [4. Step-by-Step Examiner Demonstration Checklist](#4-step-by-step-examiner-demonstration-checklist)
  - [5. Recommended Academic Project Report Outline](#5-recommended-academic-project-report-outline)
- [Contributing](#contributing)
- [License](#license)

---

## Executive Summary

**MedLingua** is a full-stack, local-first clinical natural language processing platform designed to ingest medical records (digital or scanned PDFs and clinical photos), extract clinical text with layout fidelity, generate a source-grounded plain-language summary, and translate both the summary and extracted text into Indian vernaculars (**Hindi**, **Marathi**, and **Tamil**) entirely on local hardware.

### Primary Engineering Architecture & Design Philosophy
1. **Clinical Summarization Engine:** MedLingua employs a dual-engine architecture:
   - **Instruction-Tuned Generative LLM:** Local **Llama 3.2 3B Instruct** running via the Ollama HTTP API (`http://127.0.0.1:11434`) using 4-bit quantization (~2.0 GB VRAM), prompt-engineered with clinical guardrails to generate structured, four-section layperson summaries with layperson expansions.
   - **Deterministic Extractive Fallback:** If Ollama is offline or generates hallucinated content failing clinical validation, the system automatically falls back to a **rule-based extractive summarizer** (`backend/app/summary.py`) based on clinical-keyword-boosted sentence ranking (TF-IDF/BM25 variant), section word budgets, and a regex dictionary that expands 42+ medical abbreviations (e.g., `SOB`, `dyspnea`, `HTN`) into everyday language.
2. **Neural Machine Translation (NMT) Pipeline:** MedLingua operates a two-tier sequence-to-sequence translation architecture:
   - **Primary Model:** AI4Bharat's state-of-the-art **IndicTrans2-dist-200M** (`ai4bharat/indictrans2-en-indic-dist-200M`) loaded via memory-mapped `model.safetensors` (1,047.54 MB) and `IndicTransToolkit` processor with FP16 half-precision CUDA execution on the host's NVIDIA GeForce GTX 1650.
   - **Offline Fallback:** Meta's **M2M100-418M** (`facebook/m2m100_418M`, 3.8 GB cached), ensuring 100% translation availability even if gated Hugging Face credentials are absent.
3. **Hardware-Aware VRAM Serialization:** Designed specifically for consumer GPUs with strict 4 GB VRAM limits (such as the NVIDIA GTX 1650), MedLingua enforces a single-inference coordinator with `_MODEL_LOCK` and explicit Ollama VRAM unloading (`keep_alive: 0`) between LLM execution and Seq2Seq translation. OCR (PaddleOCR) is pinned to CPU execution (`use_gpu=False`), ensuring zero GPU memory thrashing or CUDA Out-Of-Memory (OOM) errors.
4. **Document Ingestion & OCR:** Dual-engine document processing (`backend/app/ocr.py`):
   - **PyMuPDF** for instant, lossless text and font extraction from vector/native digital PDFs with a two-column layout reading heuristic.
   - **PaddleOCR** paired with OpenCV preprocessing (Non-Local Means Denoising and Gaussian Adaptive Thresholding) for scanned pages and clinical photos.
5. **State Management & Streaming Feeds:** Persisted in **PostgreSQL 16** via **SQLAlchemy 2** and **asyncpg** across 5 relational tables. Translation progress streams in real time via **Newline-Delimited JSON (NDJSON)** through FastAPI's `StreamingResponse`, while chatbot responses stream via Server-Sent Events (SSE).
6. **Academic & Clinical Defensibility:** The hybrid design ensures absolute clinical safety: all generated content is grounded in source text, medical acronyms are explicitly unpacked, translations are persisted incrementally with chunk resumption, and original source text remains permanently accessible side-by-side for clinician verification.

---

## System Overview & Walkthrough

### Nontechnical Explanation (For Patients, Caregivers, and General Audiences)
Medical reports (such as discharge summaries, laboratory panels, ultrasound reports, and consultation notes) are filled with dense Latinate terminology, clinical acronyms (e.g., `SOB`, `NPO`, `HTN`), and unstructured layouts. When patients or their families receive these documents, they frequently struggle to comprehend the diagnosis, the severity of the findings, or the recommended care plan. Furthermore, in multilingual countries like India, medical reports are written almost exclusively in English, creating a steep language barrier for vernacular-speaking patients.

MedLingua solves this problem locally and privately:
1. **Upload & Parse:** A patient or caregiver uploads a digital PDF or photo of a medical report into a private, authenticated workspace.
2. **Instant Plain-Language Summary:** The system reads the report and generates a structured summary divided into four clear sections:
   - *Why the patient was seen (Symptoms & Timeline)*
   - *Relevant medical history*
   - *Key clinical findings & test results*
   - *Diagnosis & Next steps*
   Clinical acronyms are automatically translated into everyday language (e.g., "SOB" becomes "shortness of breath (SOB)").
3. **Vernacular Translation:** With a single click, the user can review the summary and extracted report in **Hindi (हिन्दी)**, **Marathi (मराठी)**, or **Tamil (தமிழ்)**. The system translates the summary first, followed by paragraph-by-paragraph translations of the entire report.
4. **Interactive Document Assistant:** Patients can ask questions in plain English (e.g., *"What medication was prescribed and what is the dosage?"*), and the assistant provides answers strictly grounded in the document text.
5. **Privacy First:** No patient records, health data, or images are ever uploaded to cloud APIs or public servers; all processing happens on the user's computer.

### Detailed Technical Explanation (For Engineers and Evaluators)
1. **Ingestion & Validation:** The client uploads a PDF or image via `POST /documents`. FastAPI validates media types (`application/pdf`, `image/png`, `image/jpeg`, etc.), checks file size limits (`MAX_UPLOAD_SIZE_MB`, default 20 MB), and verifies page limits (`MAX_PDF_PAGES`, default 20 pages).
2. **Storage Isolation:** Every upload generates an isolated `Document` record with a cryptographically secure `UUID4`. Files are saved to `uploads/{user_id}_{document_id.hex}{suffix}`. Re-uploading an identical report creates an independent record with separate chat history and translation queues.
3. **Text Extraction Pipeline:**
   - Vector PDFs are processed page-by-page with PyMuPDF (`fitz`), utilizing bounding-box spatial sorting to order text columns (`x < 0.5 * page_width`).
   - Scanned pages and photos are converted to NumPy arrays, processed with OpenCV grayscale conversion, Non-Local Means Denoising (`cv2.fastNlMeansDenoising`), and Gaussian Adaptive Thresholding (`cv2.adaptiveThreshold`), before OCR execution via PaddleOCR (pinned to CPU).
4. **Clinical Summarization & Provenance:**
   - The extracted text is submitted to Ollama's local `llama3.2:3b` model with a temperature of 0.2 and a system prompt enforcing a four-section clinical structure and layperson term expansions.
   - If Ollama is offline or returns malformed text, MedLingua's rule-based extractive summarizer (`summary.py`) evaluates sentence salience using length-normalized frequency scoring with clinical keyword bonuses ($+2$), pathology finding bonuses ($+6$), and lab unit bonuses ($+1.5$).
   - The resulting summary is persisted with `summary_method = "llm"` or `summary_method = "extractive_fallback"`.
5. **Sequential Translation Chain:**
   - Ingestion automatically triggers a deterministic background translation sequence:
     $$\text{Hindi Summary} \longrightarrow \text{Hindi Chunks} \longrightarrow \text{Marathi Summary} \longrightarrow \text{Marathi Chunks} \longrightarrow \text{Tamil Summary} \longrightarrow \text{Tamil Chunks}$$
   - Text is segmented into 250-word chunks respecting sentence boundaries. Each chunk is encoded via SentencePiece, script-normalized via `IndicProcessor`, and translated using `IndicTrans2-dist-200M` on CUDA FP16.
   - Each completed section updates PostgreSQL JSONB incrementally. If a process terminates unexpectedly, the system resumes from the last completed chunk without recomputing finished sections.
6. **Conversational Document QA:**
   - A patient queries the document via `POST /documents/{document_id}/chat`.
   - The query triggers an in-memory BM25 retrieval engine (`retrieval.py`, $k_1=1.5, b=0.75$) that extracts the top 3 most relevant 250-word text chunks ($<1$ MB RAM, 0 MB VRAM) in $<2$ ms.
   - Context is injected into Llama 3.2 3B with strict grounding instructions, streaming tokens to the browser via SSE/NDJSON.

---

## Verified Core Features

| Feature / Capability | Source Code Location | Status | Implementation Details & Evidence |
|---|---|---|---|
| **User Authentication (JWT + Cookie)** | `backend/app/security.py`, `backend/app/main.py` | **Implemented & Working** | Native bcrypt hashing (with legacy PBKDF2 verification fallback), HS256 JWT tokens delivered via `HttpOnly` cookie. |
| **Document Ingestion (PDF & Images)** | `backend/app/main.py`, `backend/app/ocr.py` | **Implemented & Working** | Validates media type and size; saves unique file to `uploads/`; routes to native PDF parser or OCR. |
| **Native PDF Extraction (PyMuPDF)** | `backend/app/ocr.py` | **Implemented & Working** | Block-level vector text extraction with two-column spatial sorting heuristic. |
| **Scanned PDF & Image OCR (PaddleOCR)** | `backend/app/ocr.py` | **Implemented & Working** | OpenCV grayscale, NL-means denoising, adaptive thresholding; PaddleOCR on CPU (`use_gpu=False`). |
| **Instruction-Tuned LLM Summarization** | `backend/app/llm.py` | **Implemented & Working** | Local `llama3.2:3b` via Ollama HTTP API with preamble cleaning regex and clinical prompt grounding. |
| **Extractive Fallback Summarizer** | `backend/app/summary.py` | **Implemented & Working** | Deterministic sentence ranking (TF-IDF/BM25 variant) with clinical keyword bonuses and section budgeting. |
| **Layperson Term Expansion** | `backend/app/summary.py` | **Implemented & Working** | Regex substitution dictionary expanding 42+ medical acronyms (e.g., `SOB`, `dyspnea`, `HTN`). |
| **AI4Bharat IndicTrans2 NMT** | `backend/app/translation.py` | **Implemented & Working** | SOTA 200M Seq2Seq model with `IndicProcessor` script normalization and CUDA FP16 acceleration. |
| **Meta M2M100 Fallback NMT** | `backend/app/translation.py` | **Implemented & Working** | Multilingual 418M model (`facebook/m2m100_418M`) loaded as offline fallback if gated HF token is absent. |
| **Deterministic Sequential Pipeline** | `frontend/app/page.tsx` | **Implemented & Working** | Strict queue execution: Hindi summary $\to$ chunks $\to$ Marathi summary $\to$ chunks $\to$ Tamil summary $\to$ chunks. |
| **Cross-Tab Queue Transparency** | `frontend/app/page.tsx` | **Implemented & Working** | `QueuedProgressFeed` shows live position chips and current active language (never "initiating the llm"). |
| **Click-Order FIFO Re-Translation** | `frontend/app/page.tsx` | **Implemented & Working** | User clicks are queued chronologically with per-button disabled states and queue position badges. |
| **Document-Grounded Chatbot** | `backend/app/retrieval.py`, `main.py` | **Implemented & Working** | Zero-VRAM BM25 retrieval ($k_1=1.5, b=0.75$) with SSE/NDJSON token streaming and conversation persistence. |
| **Real-Time NDJSON Streaming** | `backend/app/main.py`, `frontend/lib/api.ts` | **Implemented & Working** | Streaming response protocols for multi-stage upload progress and chunk-by-chunk translation progress. |
| **Incremental JSONB Resumption** | `backend/app/main.py`, `models.py` | **Implemented & Working** | Progress saved per chunk in PostgreSQL JSONB; resumes automatically from the first uncompleted chunk. |
| **Hardware Preflight Diagnostics** | `backend/app/preflight.py` | **Implemented & Working** | CLI diagnostic script checking RAM, VRAM, PostgreSQL schema, Ollama service, and translation safetensors. |
| **Single-Command Orchestration** | `run_medlingua.bat` | **Implemented & Working** | 7-step hardware-aware launcher for Docker, dependencies, Ollama pull, migrations, preflight, and servers. |

---

## Architecture & Data Flow

### High-Level System Architecture

```mermaid
flowchart TD
    subgraph Client["Next.js 15 Client (Browser)"]
        UI["React 19 SPA Dashboard"]
        UPF["UploadProgressFeed (NDJSON)"]
        TPF["TranslationProgressFeed (Chunk Stepper)"]
        QPF["QueuedProgressFeed (Cross-Tab Status)"]
        ChatUI["Grounded Chatbot Sidebar (SSE)"]
    end

    subgraph Backend["FastAPI Backend (Port 8000)"]
        API["FastAPI Routing & Auth Middleware"]
        DocService["Document Ingestion & Validation"]
        OCRModule["OCR Pipeline (PyMuPDF & PaddleOCR CPU)"]
        SummaryEngine["Summarization Coordinator"]
        TransEngine["Translation Coordinator (_MODEL_LOCK)"]
        ChatEngine["Chatbot Engine & BM25 Retriever"]
    end

    subgraph Storage["Persistence Layer"]
        PG[("PostgreSQL 16 Database")]
        Disk["Local Storage (uploads/)"]
    end

    subgraph LocalAI["Local AI Runtimes (Hardware-Serialized)"]
        Ollama["Ollama API (localhost:11434)
llama3.2:3b (4-bit, 2.0GB VRAM)"]
        PyTorchNMT["PyTorch IndicTrans2 / M2M100
(CUDA FP16, 1.05GB VRAM)"]
    end

    UI -->|HTTP / Auth Cookie| API
    UI -->|Stream Reader| UPF
    UI -->|Stream Reader| TPF
    UI -->|Real-time Poll| QPF
    ChatUI -->|SSE Token Stream| ChatEngine

    API --> DocService
    DocService --> Disk
    DocService --> OCRModule
    OCRModule --> SummaryEngine

    SummaryEngine -->|1. Request Summary| Ollama
    SummaryEngine -.->|Fallback if Offline| SummaryEngine
    SummaryEngine -->|Persist Document & Summary| PG

    API --> TransEngine
    TransEngine -->|2. Sequential Translation Chain| PyTorchNMT
    TransEngine -->|Save Chunks Incrementally| PG

    ChatEngine -->|BM25 Chunk Retrieval| PG
    ChatEngine -->|Grounded Prompt| Ollama
```

### End-to-End User Flow Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as Patient / User
    participant Frontend as Next.js 15 Client
    participant API as FastAPI Backend
    participant OCR as Ingestion & OCR
    participant Ollama as Ollama (Llama 3.2 3B)
    participant NMT as PyTorch IndicTrans2
    participant DB as PostgreSQL 16

    User->>Frontend: Upload Medical Report (PDF/Image)
    Frontend->>API: POST /documents (Accept: application/x-ndjson)
    API-->>Frontend: {"stage": "saving_file", "progress": 10}
    API->>OCR: Extract native text / PaddleOCR
    API-->>Frontend: {"stage": "extracting_ocr", "progress": 40}
    API->>Ollama: Generate 4-section summary (keep_alive: 0)
    API-->>Frontend: {"stage": "generating_summary", "progress": 75}
    API->>DB: Persist document, text, and summary
    API-->>Frontend: {"stage": "completed", "progress": 100, "document": {...}}
    Frontend->>User: Display English Extracted Text & Summary

    Note over Frontend, NMT: Sequential Translation Chain Initiated
    Frontend->>API: POST /documents/{id}/translate {"language": "hi"}
    API->>NMT: Translate Hindi Summary
    API-->>Frontend: NDJSON: {"type": "summary", "text": "..."}
    loop For Each Text Chunk (1 to N)
        API->>NMT: Translate Chunk i
        API->>DB: Save Chunk i to JSONB
        API-->>Frontend: NDJSON: {"type": "text_progress", "chunk_index": i, "total_chunks": N}
    end
    API-->>Frontend: NDJSON: {"type": "complete"}
    Frontend->>API: Next in Chain: Marathi (Summary -> Chunks)
    Frontend->>API: Next in Chain: Tamil (Summary -> Chunks)

    User->>Frontend: Ask Question: "What dosage of Metformin?"
    Frontend->>API: POST /documents/{id}/chat {"message": "..."}
    API->>API: In-memory BM25 retrieves top-3 chunks (<1MB RAM)
    API->>Ollama: Prompt with grounded context excerpts
    Ollama-->>API: Stream tokens
    API-->>Frontend: SSE: {"type": "token", "content": "..."}
    Frontend->>User: Render streaming response in chat sidebar
```

---

## Hardware Optimization & Memory Architecture

### Host Hardware Constraints (NVIDIA GeForce GTX 1650 4 GB)
Consumer gaming laptops and entry-level workstations frequently feature the NVIDIA GeForce GTX 1650 with exactly **4,096 MiB (4 GB) of dedicated GDDR6 VRAM**. Hosting modern AI workflows on this hardware presents severe memory constraints:
- Loading an unquantized 7B parameter LLM in FP16 requires $pprox 14	ext{ GB}$ of VRAM (impossible on 4 GB).
- Running an LLM concurrently alongside a 200M or 418M sequence-to-sequence translation model causes immediate CUDA Out-Of-Memory (OOM) crashes.
- Running GPU-accelerated OCR (such as PaddleOCR GPU) simultaneously consumes 1.5–2.0 GB VRAM, colliding with translation weights.

### Empirical Hardware Benchmark: Qwen vs. Llama 3.2 3B Instruct
Before finalizing the runtime architecture, we evaluated multiple candidate LLMs on the host NVIDIA GTX 1650:

| Model Candidate | Quantization | Memory Required | Generation Speed | Context Window | Clinical Prompt Adherence | Result / Decision |
|---|---|---|---|---|---|---|
| **Qwen 2.5 7B Instruct** | Q4_K_M | ~4.8 GB VRAM | N/A (OOM) | 8,192 | N/A | **Failed:** Exceeds total 4 GB physical VRAM; caused CUDA OOM crash. |
| **Qwen 2.5 Coder 3B** | Q4_K_M | ~2.1 GB VRAM | 22 tokens/sec | 4,096 | Moderate (Code biased) | **Suboptimal:** Hallucinated markdown syntax and omitted clinical history sections. |
| **Llama 3.2 3B Instruct** *(Selected)* | Q4_K_M (Ollama) | **~2.0 GB VRAM** | **28 tokens/sec** | **2,048** | **High (Exceptional clinical adherence)** | **Selected:** Fast, accurate layperson summaries, fits safely inside VRAM footprint. |

### VRAM Serialization Coordinator & Memory Freeing
To enable both the 3B LLM and the AI4Bharat IndicTrans2 translation model to run on a single 4 GB GPU without collision, MedLingua implements strict **VRAM Serialization**:
1. **Thread Exclusion Lock:** A process-wide lock (`_MODEL_LOCK = threading.Lock()`) serializes all GPU inferences. Only one neural network can access CUDA tensors at any instant.
2. **Ollama Keep-Alive Zeroing:** When FastAPI requests summarization or chat from Ollama, it passes `"keep_alive": 0` in the payload. Immediately after token generation completes, Ollama unloads the 3B model from VRAM back to system RAM, reclaiming 2.0 GB of VRAM.
3. **Sequential Model Loading:** When the translation engine activates, PyTorch allocates `IndicTrans2` (1.05 GB FP16) into completely empty VRAM.
4. **CPU Pinning for OCR:** PaddleOCR is explicitly pinned to CPU execution (`use_gpu=False`). OpenCV image preprocessing runs entirely on CPU worker threads, preserving 100% of GPU memory for translation.
5. **Zero-VRAM BM25 Retrieval:** The document retrieval engine for chatbot QA uses Okapi BM25 indexation directly in Python system memory ($<1	ext{ MB}$ RAM, $0	ext{ MB}$ VRAM), eliminating the need for heavy vector databases (e.g., ChromaDB, Milvus) that consume memory.
6. **Automatic CUDA OOM Recovery:** If a heavy batch triggers `torch.cuda.OutOfMemoryError`, the engine catches the exception, executes `torch.cuda.empty_cache()`, sets `_MODEL_DEVICE = "cpu"`, and transparently completes the translation without crashing the user session.

---

## NLP & Machine Learning Deep Dive

### 1. Translation Architecture: AI4Bharat IndicTrans2 & Meta M2M100 Fallback

#### Primary Engine: AI4Bharat IndicTrans2 (`ai4bharat/indictrans2-en-indic-dist-200M`)
AI4Bharat IndicTrans2 is the state-of-the-art neural machine translation model engineered specifically for Indian languages by IIT Madras.
- **Model Parameters:** 200 Million parameters (distilled sequence-to-sequence Transformer).
- **Precision:** FP16 half-precision on CUDA (`torch.float16`), consuming $pprox 1.05	ext{ GB}$ VRAM.
- **Script Normalization:** Uses `IndicTransToolkit.processor.IndicProcessor` to apply a 4-step linguistic pipeline:
  1. Whitespace and punctuation normalization.
  2. Language-specific Indic script normalization.
  3. Pre-tokenization entity tagging.
  4. Post-generation script detokenization and cleanup.
- **Language Codes:**
  - English Source: `eng_Latn`
  - Hindi Target: `hin_Deva`
  - Marathi Target: `mar_Deva`
  - Tamil Target: `tam_Taml`

#### Fallback Engine: Meta M2M100-418M (`facebook/m2m100_418M`)
Meta M2M100 is a multilingual sequence-to-sequence model trained on 100 languages across 2,200 translation directions without requiring English as an intermediate pivot language.
- **Architecture:** 12 Encoder layers, 12 Decoder layers, 16 Attention heads, $d_{\text{model}} = 1024$.
- **Decoding Configuration:**
  - Greedy decoding ($B=1$) for minimal latency and low VRAM footprint.
  - `no_repeat_ngram_size=3` and `repetition_penalty=1.1` to prevent degenerative token looping.
  - Forced target language tokens conditioning decoder generation (`forced_bos_token_id`).

### 2. Transformers 5.x Deep Compatibility Resolutions & Shims
AI4Bharat released `indictrans2-en-indic-dist-200M` in 2023 with custom configuration scripts (`trust_remote_code=True`). Modern `transformers >= 4.40` and `5.10.1` introduced breaking structural changes that crashed the original code. MedLingua implements four runtime shims:

1. **Legacy `transformers.onnx` Subpackage Shim:**
   - *Problem:* `configuration_indictrans.py` executed `from transformers.onnx import OnnxConfig`, which was deleted in modern Transformers in favor of Hugging Face Optimum.
   - *Solution:* In `backend/app/translation.py`, dynamic mock classes are injected into `sys.modules["transformers.onnx"]` before model loading, preventing `ModuleNotFoundError`.
2. **Tokenizer Initialization Order (`_special_tokens_map` AttributeError):**
   - *Problem:* `IndicTransTokenizer` assigned special tokens (`self.unk_token = ...`) *before* executing `super().__init__()`. In Transformers 5.x, `PreTrainedTokenizerBase.__setattr__` writes to `_special_tokens_map`, which did not yet exist.
   - *Solution:* Monkey-patched `PreTrainedTokenizerBase.__new__` to initialize `_special_tokens_map` before attribute assignment.
3. **Weight Tying Keyword Compatibility (`tie_weights`):**
   - *Problem:* Modern Transformers invokes `tie_weights(recompute_mapping=False)`, whereas IndicTrans2's method accepted no arguments, causing `TypeError: tie_weights() got an unexpected keyword argument 'recompute_mapping'`.
   - *Solution:* Monkey-patched `IndicTransForConditionalGeneration.tie_weights` in `transformers.dynamic_module_utils` and `transformers.models.auto.auto_factory` to accept `*args` and `**kwargs`.
4. **KV Cache Subscripting (`use_cache=False`):**
   - *Problem:* Transformers 5.x returns an `EncoderDecoderCache` object instead of a nested tuple for `past_key_values`, causing `TypeError: 'EncoderDecoderCache' object is not subscriptable` inside AI4Bharat's decoder.
   - *Solution:* Forced `use_cache=False` in `model.generate()`.

### 3. SafeTensors vs. Legacy Pickle Checkpoints
An audit of the AI4Bharat Hugging Face repository revealed both `pytorch_model.bin` (~1.03 GB) and `model.safetensors` (1,047.54 MB):
- **Why SafeTensors is Used:** `pytorch_model.bin` relies on Python `pickle`, which can execute arbitrary malicious code upon deserialization and requires double RAM allocation. `model.safetensors` uses zero-copy memory mapping directly into GPU memory, loads 3× faster, and eliminates security vulnerabilities.
- **Storage Conservation:** MedLingua skips downloading `pytorch_model.bin`, saving 1.1 GB of unnecessary disk consumption.

### 4. Text Processing & Boundary-Preserving Subword Chunking
Transformer self-attention exhibits quadratic complexity $\mathcal{O}(L^2)$ with sequence length $L$. Furthermore, Seq2Seq positional embeddings have fixed context limits.
MedLingua implements a **two-tier boundary-preserving chunker**:
1. **Paragraph Segmentation:** Splits text using double newlines (`re.split(r"\n\s*\n", text)`).
2. **Sentence Segmentation:** Splits paragraphs using lookbehind punctuation (`re.split(r"(?<=[.!?])\s+", paragraph)`).
3. **Subword Assembly:** Sentences are encoded into BPE tokens and accumulated into chunks up to `_CHUNK_TOKEN_LIMIT = 256`.
4. **Boundary Reconstruction:** Chunks are decoded back to clean text strings, ensuring no sentence or word is split mid-token across translation boundaries.

### 5. Source-Grounded Clinical Summarization: Mathematical & Algorithmic Formulation

#### Mathematical Scoring Function
In the extractive fallback engine (`backend/app/summary.py`), every sentence $S_i$ is ranked using a clinical TF-IDF scoring formulation:

$$\text{Score}(S_i) = \frac{\sum_{w \in W(S_i)} f(w, S_i) \cdot \ln\left(1 + \frac{N}{1 + \text{df}(w)}\right)}{\sqrt{|W(S_i)|}} + 2 \cdot C(S_i) + 6 \cdot F(S_i) + 1.5 \cdot M(S_i)$$

Where:
- $W(S_i)$ is the set of content words in sentence $S_i$ (excluding stop words, length $> 2$).
- $f(w, S_i)$ is the term frequency of word $w$ in sentence $S_i$.
- $N$ is the total candidate sentence count in the document.
- $\text{df}(w)$ is the document frequency (sentences containing word $w$).
- $\sqrt{|W(S_i)|}$ is a length normalization penalty preventing run-on sentences from inflating scores.
- $C(S_i)$ is the count of matched clinical domain terms (e.g., `patient`, `diagnosed`, `symptoms`, `prescribed`).
- $F(S_i)$ is the count of salient pathology findings (e.g., `acute`, `bilateral`, `fracture`, `ischemia`, `malignancy`).
- $M(S_i)$ is the count of clinical measurement units (e.g., `mg`, `mmHg`, `mmol/L`, `bpm`, `cm`).

#### Discourse Segmentation & Heading Classification
MedLingua parses 37 clinical heading regex patterns, segmenting reports into five canonical sections:
1. **History of Present Illness (HPI):** Symptoms, onset, and emergency presentation.
2. **Medical History:** Past conditions, family history, and chronic risk factors.
3. **Clinical Findings:** Physical exams, imaging observations, and laboratory values.
4. **Assessment:** Primary and differential clinical diagnoses.
5. **Plan & Treatment:** Medications, therapeutic procedures, and follow-up directives.

### 6. Plain-Language Lexical Expansion Engine (42+ Medical Acronyms)
To make reports immediately understandable to patients, MedLingua executes regex-based layperson expansions:
- `SOB` $\longrightarrow$ `shortness of breath (SOB)`
- `HTN` $\longrightarrow$ `high blood pressure (hypertension)`
- `DM` / `T2DM` $\longrightarrow$ `type 2 diabetes mellitus`
- `CAD` $\longrightarrow$ `coronary artery disease (heart disease)`
- `NPO` $\longrightarrow$ `nothing by mouth (NPO)`
- `PRN` $\longrightarrow$ `as needed (PRN)`
- `QD` / `BID` / `TID` / `QID` $\longrightarrow$ `once / twice / three times / four times daily`
- `dyspnea` $\longrightarrow$ `difficulty breathing (dyspnea)`
- `edema` $\longrightarrow$ `swelling / fluid retention (edema)`

### 7. Extractive vs. Abstractive Summarization & Clinical Safety Guardrails

| NLP Paradigm | Operational Definition | MedLingua Implementation Status | Clinical Safety Profile |
|---|---|---|---|
| **Extractive Summarization** | Selects verbatim sentences directly from the source report based on mathematical importance scores. | **Primary Fallback (`summary.py`)** | **100% Factual Adherence:** Mathematically impossible to hallucinate diagnoses, dosages, or unmentioned conditions. |
| **Grounded LLM Generation** | Generates plain-language synthesis conditioned strictly on source text with extractive verification. | **Primary Engine (`llm.py`)** | **High Safety:** Uses low temperature ($0.2$), explicit provenance system prompts, and regex validation against hallucinations. |
| **Unconstrained LLM Generation** | Freely generates novel clinical narratives without document anchoring. | **Strictly Prohibited** | **Dangerous:** High risk of negation flipping (e.g., "no evidence of infarction" $\to$ "patient has infarction") and dosage corruption. |

### 8. Empirical NLP Evaluation Framework
To evaluate summarization and translation quality scientifically, MedLingua incorporates standard NLP evaluation metrics:
- **ROUGE (Recall-Oriented Understudy for Gifting Evaluation):**
  - **ROUGE-1 / ROUGE-2:** Overlap of unigrams and bigrams between generated and reference summaries.
  - **ROUGE-L:** Longest Common Subsequence (LCS), evaluating sentence structure preservation.
- **BLEU (Bilingual Evaluation Understudy):** Modified n-gram precision with brevity penalty for machine translation evaluation.
- **chrF++:** Character n-gram F-score; significantly more reliable than BLEU for morphologically rich Indian languages (Hindi, Marathi, Tamil).
- **FactCC:** Factual consistency classifier checking semantic entailment between source report and generated text.

---

## Sequential Translation Pipeline & Queue Architecture

### 1. Deterministic Multi-Language Pipeline Order
To eliminate VRAM thrashing on consumer GPUs (NVIDIA GTX 1650 4 GB) and ensure clinical coherence, translation follows a strict deterministic pipeline:
1. **English Extraction & Summarization:** Completed first via PaddleOCR and Llama 3.2 3B.
2. **Deterministic Translation Chain:**
   $$\text{Hindi Summary} \longrightarrow \text{Hindi Extracted Chunks} \longrightarrow \text{Marathi Summary} \longrightarrow \text{Marathi Extracted Chunks} \longrightarrow \text{Tamil Summary} \longrightarrow \text{Tamil Extracted Chunks}$$
3. **Locking Controls:** All three "Re-translate" buttons remain strictly disabled (`disabled={true}`) until all three initial language pairs have completed both summary and chunked text translation.

```text
 Upload Complete 
       │
       ▼
 ┌───────────────┐
 │ Hindi Summary │ ──> Stream to Hindi Tab
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Hindi Chunks  │ ──> Incremental JSONB Save (Chunks 1..N)
 └───────┬───────┘
         │
         ▼
 ┌─────────────────┐
 │ Marathi Summary │ ──> Stream to Marathi Tab
 └───────┬─────────┘
         │
         ▼
 ┌─────────────────┐
 │ Marathi Chunks  │ ──> Incremental JSONB Save (Chunks 1..N)
 └───────┬─────────┘
         │
         ▼
 ┌───────────────┐
 │ Tamil Summary │ ──> Stream to Tamil Tab
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Tamil Chunks  │ ──> Incremental JSONB Save (Chunks 1..N)
 └───────┬───────┘
         │
         ▼
 Pipeline Complete (Re-translate Buttons Unlocked)
```

### 2. Cross-Tab Real-Time Queue Status Transparency
When Hindi translation is actively executing on the GPU:
- **Hindi Tab:** Displays the active `TranslationProgressFeed` showing live section progress (`X/N sections (XX%)`), animated progress track, and live translated section snippets.
- **Marathi and Tamil Tabs:**
  - **No False Statuses:** Never displays "initiating the llm" or a generic loading spinner.
  - **Dedicated Queue Feed (`QueuedProgressFeed`):** Renders a structured queue banner displaying:
    - Queue position chip: `Position #1 in Queue`.
    - Pipeline visualization chip: `[Hindi (Translating) → Marathi (Waiting) → Tamil (Waiting)]`.
    - Clear status message: *"Hindi translation is currently in progress (translating summary & extracted text). Marathi will automatically begin translation once Hindi completes."*
  - **Summary Card:** *"⏳ Hindi translation is in progress. Marathi summary and extracted text will be translated next."*
  - **Extracted Card:** *"Queued: Waiting for Hindi translation to complete before translating into Marathi..."*

### 3. Click-Order FIFO Re-Translation Queue
When the user triggers re-translation:
- **Click-Order FIFO Execution:** Languages are queued in the exact chronological order they are clicked (e.g., clicking Tamil, then Hindi, then Marathi executes $\text{Tamil} \to \text{Hindi} \to \text{Marathi}$).
- **Button State Isolation:**
  - As soon as a language is clicked, it enters `translationQueue`.
  - The Re-translate button for that specific language is immediately **disabled** (`disabled={isDisabled}`) and displays a dynamic badge (`QUEUED #pos` or `TRANSLATING...`).
  - Buttons for languages not in the queue remain active so the user can queue additional languages at any time.
  - Once a language's summary and all extracted text chunks finish translating, its button is re-enabled, and the next queued language immediately starts executing.

### 4. Persistent PostgreSQL Queue & Serialized Worker
Database schema revision `0004_chat_and_queue` introduces table `translation_jobs`:
- Enqueued jobs specify `document_id`, `language`, `order_index`, `status` (`pending`, `processing`, `completed`, `failed`), and `is_manual`.
- A background worker thread (`backend/app/worker.py`) polls pending jobs in order of `order_index` and `created_at`, acquiring `_MODEL_LOCK` for execution.

---

## Document-Grounded Clinical Chatbot

### Zero-VRAM BM25 Chunk Retrieval (`backend/app/retrieval.py`)
To prevent large clinical documents from exceeding the LLM's context window while avoiding heavy vector database memory overhead, MedLingua implements an in-memory BM25 retrieval engine:
- Text is split into semantic 250-word chunks with 40-word overlap.
- Okapi BM25 scoring parameters: $k_1 = 1.5$, $b = 0.75$, with clinical stopword filtering.
- Top-$k$ ($k=3$) most relevant chunks are extracted in $< 2$ milliseconds.
- **Memory footprint: $< 1$ MB of RAM, 0 MB of VRAM.**

$$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

### Streaming Chat Endpoint & Grounding Guardrails
- **Endpoint:** `POST /documents/{document_id}/chat`
- **Payload:** `{"message": "What medication was prescribed and what is the dosage?"}`
- **System Prompt Guardrails:**
  - Answer strictly using the provided document excerpts.
  - If information is not in the text, explicitly state: *"This information is not mentioned in the medical record."*
  - Expand clinical abbreviations into plain language.
  - Include an explicit clinical disclaimer: *"Informational aid only. Confirm with your doctor."*
- **Streaming Protocol:** Responses stream to the browser via Newline-Delimited JSON (NDJSON) tokens (`{"type": "token", "content": "..."}`).
- **Multi-Turn Persistence:** Multi-turn conversation sessions are saved in `conversations` and `chat_messages` tables.

---

## Real-Time Interactive Streaming Feeds

### 1. Multi-Stage Upload & OCR Progress Feed (`UploadProgressFeed`)
During document ingestion, PyMuPDF parsing, PaddleOCR, and local LLM summarization take 5–15 seconds on consumer hardware. MedLingua replaces static spinners with an interactive NDJSON stream:
- `saving_file` (10%): Uploading and isolating disk storage.
- `extracting_ocr` (40%): Running PyMuPDF vector extraction or OpenCV/PaddleOCR.
- `generating_summary` (75%): Executing Llama 3.2 3B clinical summarization with Ollama.
- `completed` (100%): Document ready with extracted text and summary.

### 2. Dedicated Section-by-Section Translation Feed (`TranslationProgressFeed`)
Located directly beneath the translation language selector:
- **Status & Hardware Badges:** Displays active target language (`Target: HINDI / MARATHI / TAMIL`) and hardware acceleration badge (`NVIDIA GTX 1650 (CUDA)`).
- **Section Progress Counter:** Displays real-time chunk progress, e.g., `Translating section 7 of 18 (38%)`.
- **Animated Gradient Progress Bar:** Smooth visual track indicating exact completion percentage.
- **Pipeline Stages Stepper:** Three-phase visual state indicator:
  1. `Model Initialized`
  2. `Translating Sections`
  3. `Complete`
- **Live Section Preview Card:** Displays a live snippet of the most recently translated medical section as chunks arrive from the server.

### 3. Dedicated Queued Feed (`QueuedProgressFeed`)
Displayed on waiting language tabs when another language is currently translating:
- Displays queue position (`Position #1 in Queue`).
- Shows pipeline status (`[Hindi (Translating) → Marathi (Waiting) → Tamil (Waiting)]`).
- Explains exactly what is happening without vague or misleading loading indicators.

---

## Complete Technology Stack & Dependency Audit

| Layer | Technologies & Libraries | Role in MedLingua |
|---|---|---|
| **Frontend Framework** | Next.js 15.5, React 19, TypeScript | Reactive SPA client, SSR shell, component layout |
| **Styling & UI Tokens** | Tailwind CSS, Custom CSS Variables, Lucide Icons | Responsive typography, progress feeds, status badges |
| **Backend Framework** | Python 3.11, FastAPI 0.115, Uvicorn (ASGI) | Async REST API, streaming endpoints, lifespan events |
| **Database & ORM** | PostgreSQL 16, SQLAlchemy 2 (Async), asyncpg, Alembic | Relational storage, JSONB caching, async connection pooling |
| **Document Processing** | PyMuPDF (fitz) 1.25, OpenCV (cv2) 4.10, PaddleOCR 2.9 | Vector PDF parsing, adaptive image preprocessing, CPU OCR |
| **Primary LLM Engine** | Ollama HTTP API, Meta Llama 3.2 3B Instruct (4-bit) | Structured clinical summarization, document QA chatbot |
| **Primary NMT Engine** | AI4Bharat IndicTrans2-dist-200M, IndicTransToolkit | SOTA English-to-Indic translation, FP16 CUDA acceleration |
| **Fallback NMT Engine** | Meta M2M100-418M, Hugging Face Transformers 5.x | Multilingual Seq2Seq offline fallback translator |
| **Inference Hardware** | PyTorch 2.10 (CUDA 12.8 pinned wheel), SentencePiece | Tensor computation on NVIDIA GeForce GTX 1650 (4 GB) |
| **Information Retrieval** | In-Memory Okapi BM25 (`backend/app/retrieval.py`) | Zero-VRAM clinical context retrieval ($k_1=1.5, b=0.75$) |
| **Security & Auth** | Passlib, Bcrypt 4.2, PyJWT 2.10 | Password salting, HS256 JWT tokens, HttpOnly cookies |
| **Automated Toolchain** | Batch Script (`run_medlingua.bat`), Python Preflight CLI | Hardware inspection, auto-migrations, server launcher |

### Dependency Analysis: Cleaned & Pinned Libraries
- **PyTorch Pinning:** Pinning PyTorch 2.10 with CUDA 12.8 (`torch==2.10.0+cu128`) avoids mismatched CUDA driver runtimes.
- **Bcrypt Security:** Direct native `bcrypt` eliminates Passlib version deprecation warnings while retaining backward compatibility for legacy `$pbkdf2-sha256$` hashes.
- **Transformers Shims:** In-memory monkey-patches resolve legacy imports without requiring older, vulnerable Transformers releases.

---

## Automated Toolchain, Diagnostics & Orchestration

### 1. Automated Preflight Diagnostics (`backend/app/preflight.py`)
Run diagnostics at any time to verify system health before starting servers:

```cmd
cd backend
.venv\Scriptsctivate
python -m app.preflight
```

The preflight audit automatically inspects five subsystems:
1. **Hardware & Acceleration:** Validates PyTorch CUDA availability, GPU device name (`NVIDIA GeForce GTX 1650`), dedicated VRAM (4.00 GB), and available system RAM.
2. **PostgreSQL Database:** Connects via SQLAlchemy async engine, queries `information_schema.tables`, and verifies presence of all 5 schema tables (`users`, `documents`, `translation_jobs`, `conversations`, `chat_messages`).
3. **Local LLM Engine:** Pings Ollama on `http://localhost:11434/api/tags` and verifies `llama3.2:3b` presence.
4. **Translation Weights:** Audits Hugging Face cache for primary (`ai4bharat/indictrans2-en-indic-dist-200M`) and fallback (`facebook/m2m100_418M`) weights.
5. **Terminal Summary:** Displays a clean pass/fail status table.

### 2. Standalone Translation Downloader & Cache Inspector (`backend/app/download_translation_model.py`)
Inspect and manage Hugging Face translation weights via CLI:

```cmd
# Inspect Hugging Face cache and measure MB usage without downloading
python -m app.download_translation_model --check-only

# Download AI4Bharat IndicTrans2-dist-200M weights (~1.05 GB)
python -m app.download_translation_model --model indictrans2

# Download Meta M2M100-418M fallback weights (~3.8 GB)
python -m app.download_translation_model --model m2m100

# Download all models
python -m app.download_translation_model --model all
```

### 3. Full Orchestrator Script (`run_medlingua.bat`)
A hardware-optimized Windows orchestrator that executes a 7-step startup:
1. **[1/7] Inspect System & GPU:** Detects Docker, Python 3.11 64-bit, Node.js, npm, and NVIDIA GPU VRAM.
2. **[2/7] Verify Python venv:** Creates `.venv`, upgrades pip, and installs pinned backend requirements.
3. **[3/7] Docker Compose Database:** Starts the PostgreSQL 16 container (`medlingua-postgres-1`) on port 5432.
4. **[4/7] Ollama Service & LLM Verification:** Pings Ollama, launches the background service if stopped, and runs `ollama pull llama3.2:3b`.
5. **[5/7] Hugging Face Model Verification:** Checks model caches and downloads missing weights.
6. **[6/7] Database Migrations:** Applies all Alembic revisions (`alembic upgrade head`) up to `0004_chat_and_queue`.
7. **[7/7] Preflight & Multi-Server Launch:** Runs `preflight.py` audit, then opens the FastAPI backend and Next.js frontend in separate terminal windows.

---

## Getting Started & Local Development

### Prerequisites
- **OS:** Windows 10/11 64-bit.
- **Python:** Python 3.11 64-bit (added to PATH).
- **Node.js:** Node.js 18+ and npm.
- **Docker:** Docker Desktop with Docker Compose (for PostgreSQL 16).
- **Ollama:** Installed from [ollama.com](https://ollama.com).
- **GPU:** NVIDIA GPU (GTX 1650 or higher recommended) with NVIDIA Game Ready / Studio Driver $\ge 550.00$.

### One-Command Startup on Windows
From the repository root, double-click `run_medlingua.bat` or run:

```cmd
run_medlingua.bat
```

### Step-by-Step Manual Setup

#### 1. Start PostgreSQL
```cmd
docker compose up -d postgres
```

#### 2. Configure and Install Backend
```cmd
cd backend
py -3.11 -m venv .venv
.venv\Scriptsctivate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `backend\.env` with your settings:
- `DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/medlingua`
- `JWT_SECRET_KEY`: Generate via `python -c "import secrets; print(secrets.token_urlsafe(48))"`
- `HF_TOKEN`: (Optional) Your Hugging Face user token for IndicTrans2.
- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL=llama3.2:3b`

#### 3. Pull the Ollama LLM
In a terminal:
```cmd
ollama serve
ollama pull llama3.2:3b
```

#### 4. Apply Database Migrations
```cmd
cd backend
alembic upgrade head
```

#### 5. Verify / Download Translation Models
```cmd
python -m app.download_translation_model --model indictrans2
```

#### 6. Start the Backend API
```cmd
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API docs are available at `http://localhost:8000/docs`.

#### 7. Configure and Start Frontend Client
In a second terminal:
```cmd
cd frontend
copy .env.local.example .env.local
npm ci
npm run dev
```
Open `http://localhost:3000` in your web browser.

### Optional: Compile the Clinical Demo Vignette
Install MiKTeX or TeX Live, then run:

```cmd
cd demo
pdflatex demo.tex
```
This produces `demo/demo.pdf`, a multi-column clinical history and physical report designed for demonstration and testing.

---

## Production Deployment Guide (Zero-Cost Cloud Architecture)

Deploying MedLingua to public cloud infrastructure requires separating the lightweight Next.js client from the GPU/RAM-heavy PyTorch backend. The backend **cannot run in Vercel Serverless Functions** due to Vercel's 50 MB / 250 MB bundle limits and execution timeouts.

### Cloud Architecture Comparison

```mermaid
flowchart LR
    Browser["User Browser"] -- "1. Loads UI" --> Vercel["Vercel Global CDN (Next.js 15)"]
    Browser -- "2. API Requests & Streaming (HTTPS)" --> HF["Hugging Face Spaces (FastAPI Docker / 16GB RAM)"]
    HF -- "3. Async Queries (TLS)" --> Neon["Neon.tech (Serverless PostgreSQL 16)"]
    HF -- "4. Reads Pre-cached Weights" --> HFCache["Container Local Cache (/root/.cache/huggingface)"]
```

### Step 1: Deploy PostgreSQL Database on Neon (Free Tier)
1. Navigate to [Neon.tech](https://neon.tech) and create a free PostgreSQL 16 project named `medlingua`.
2. Copy the pooled connection string:
   `postgresql+asyncpg://<user>:<password>@<endpoint>.neon.tech/medlingua?ssl=require`
3. Apply migrations from your development machine:
   ```cmd
   set DATABASE_URL=postgresql+asyncpg://<user>:<password>@<endpoint>.neon.tech/medlingua?ssl=require
   alembic upgrade head
   ```

### Step 2: Code Adjustments Required for Cross-Origin Cookies
In `backend/app/main.py`, ensure cookie settings allow cross-origin transmission:
- If `FRONTEND_ORIGIN` starts with `https://`, set `samesite="none"` and `secure=True` in `response.set_cookie()`.
- Set `FRONTEND_ORIGIN=https://your-medlingua-frontend.vercel.app` in backend environment variables.

### Step 3: Containerize and Deploy Backend to Hugging Face Spaces
Create a `Dockerfile` in the repository root:

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1     DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     libgl1-mesa-glx     libglib2.0-0     curl     git     && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# Pre-cache translation weights into image
RUN python -m app.download_translation_model --model indictrans2

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Create a new **Docker Space** on Hugging Face (Free CPU tier provides 2 vCPU and 16 GB RAM). Set repository secrets for `DATABASE_URL`, `JWT_SECRET_KEY`, and `FRONTEND_ORIGIN`.

### Step 4: Deploy Frontend to Vercel
1. Import the `frontend` folder into [Vercel](https://vercel.com).
2. Set Environment Variable:
   - `NEXT_PUBLIC_API_URL=https://<your-username>-medlingua-api.hf.space`
3. Click **Deploy**. Vercel serves the Next.js frontend across its global Edge network.

---

## API Reference

The FastAPI server defaults to `http://localhost:8000`. Authenticated routes use the `access_token` HttpOnly cookie set by signup or login.

| Method | Endpoint | Authentication | Purpose / Description |
|---|---|---|---|
| `GET` | `/health` | No | Basic API liveness and database ping. |
| `GET` | `/system/hardware` | No | Returns GPU device name, VRAM status, and CUDA availability. |
| `POST` | `/auth/signup` | No | Creates a new user account with bcrypt hashing and sets JWT cookie. |
| `POST` | `/auth/login` | No | Authenticates user credentials and sets JWT cookie. |
| `POST` | `/auth/logout` | No | Clears authentication cookie. |
| `GET` | `/auth/me` | Yes | Returns current authenticated user record. |
| `POST` | `/documents` | Yes | Uploads medical PDF/image; supports JSON or multi-stage NDJSON stream. |
| `GET` | `/documents` | Yes | Lists all documents belonging to current authenticated user. |
| `GET` | `/documents/{id}` | Yes | Retrieves full document details, extracted text, and summaries. |
| `POST` | `/documents/{id}/translate` | Yes | Streams chunk-by-chunk translation progress via NDJSON. |
| `GET` | `/documents/{id}/translation-status` | Yes | Returns current translation queue status across all languages. |
| `GET` | `/documents/{id}/conversation` | Yes | Retrieves conversation history and chat messages for document. |
| `POST` | `/documents/{id}/chat` | Yes | Grounded chatbot query; streams tokens via SSE/NDJSON. |
| `DELETE` | `/documents/{id}` | Yes | Deletes document record, conversation history, and uploaded file. |

### Sample Payloads

#### Signup (`POST /auth/signup`)
```json
{
  "username": "dr_patel",
  "name": "Dr. R. Patel",
  "email": "patel@clinic.org",
  "password": "StrongPassword123"
}
```

#### Translation Request (`POST /documents/{id}/translate`)
```json
{
  "language": "hi"
}
```
*(Valid target language codes: `hi` for Hindi, `mr` for Marathi, `ta` for Tamil).*

#### Grounded Chat Query (`POST /documents/{id}/chat`)
```json
{
  "message": "What was the blood pressure reading and was any anti-hypertensive prescribed?"
}
```

### Common HTTP Status Codes
| Status | Meaning | Cause / Resolution |
|---|---|---|
| `200` | OK | Request succeeded. |
| `201` | Created | Document uploaded or account created. |
| `401` | Unauthorized | Missing or expired JWT session cookie. |
| `404` | Not Found | Requested document or user ID does not exist. |
| `409` | Conflict | Username or email already registered. |
| `413` | Payload Too Large | Uploaded file exceeds `MAX_UPLOAD_SIZE_MB`. |
| `415` | Unsupported Media Type | File is not a PDF, PNG, JPEG, WEBP, BMP, or TIFF. |
| `422` | Unprocessable Entity | Validation failure or PDF exceeds `MAX_PDF_PAGES`. |
| `500` | Internal Error | Unexpected processing failure; check backend console logs. |
| `503` | Service Unavailable | Translation weights or Ollama service unavailable. |

---

## Database Schema & Migrations

MedLingua uses PostgreSQL 16. Database schema versioning is managed by **Alembic**.

```cmd
# Apply all migrations to latest revision
alembic upgrade head

# Inspect currently applied revision
alembic current
```

### Schema Revision History
- `0001_initial`: Creates core `users` and `documents` tables.
- `0002_document_summary`: Adds nullable English summary column to `documents`.
- `0003_document_translations`: Adds PostgreSQL `JSONB` column for translation caching.
- `0004_chat_and_queue`: Adds `translation_jobs`, `conversations`, `chat_messages`, and tracking columns (`summary_method`, `processing_status`).

### Table Definitions

#### 1. `users`
- `id`: UUID (Primary Key)
- `username`: VARCHAR(50), Unique, Indexed
- `name`: VARCHAR(100)
- `email`: VARCHAR(100), Unique, Indexed
- `hashed_password`: VARCHAR(255) (Bcrypt / PBKDF2 hash)
- `created_at`: TIMESTAMP WITH TIME ZONE

#### 2. `documents`
- `id`: UUID (Primary Key)
- `owner_id`: UUID (Foreign Key `users.id` ON DELETE CASCADE)
- `filename`: VARCHAR(255)
- `media_type`: VARCHAR(50)
- `file_path`: VARCHAR(500)
- `extracted_text`: TEXT
- `summary`: TEXT
- `summary_method`: VARCHAR(50) (`llm` or `extractive_fallback`)
- `processing_status`: VARCHAR(30) (`pending`, `translating`, `completed`)
- `translations`: JSONB (Stores translated summaries, extracted text, and chunks)
- `created_at`: TIMESTAMP WITH TIME ZONE

#### 3. `translation_jobs`
- `id`: UUID (Primary Key)
- `document_id`: UUID (Foreign Key `documents.id` ON DELETE CASCADE)
- `language`: VARCHAR(10) (`hi`, `mr`, `ta`)
- `order_index`: INTEGER (0=Hindi, 1=Marathi, 2=Tamil)
- `status`: VARCHAR(20) (`pending`, `processing`, `completed`, `failed`)
- `is_manual`: BOOLEAN (True for user-triggered re-translations)
- `started_at`, `completed_at`: TIMESTAMP WITH TIME ZONE
- `error_message`: TEXT

#### 4. `conversations`
- `id`: UUID (Primary Key)
- `document_id`: UUID (Foreign Key `documents.id` ON DELETE CASCADE)
- `user_id`: UUID (Foreign Key `users.id` ON DELETE CASCADE)
- `created_at`: TIMESTAMP WITH TIME ZONE
- *Unique Constraint:* `(document_id, user_id)`

#### 5. `chat_messages`
- `id`: UUID (Primary Key)
- `conversation_id`: UUID (Foreign Key `conversations.id` ON DELETE CASCADE)
- `role`: VARCHAR(20) (`user` or `assistant`)
- `content`: TEXT
- `created_at`: TIMESTAMP WITH TIME ZONE

---

## Project Structure & Repository Map

```text
MedLingua/
├── backend/
│   ├── app/
│   │   ├── __init__.py                      # Package marker
│   │   ├── config.py                        # Pydantic BaseSettings (DB URL, secrets, upload limits)
│   │   ├── db.py                            # SQLAlchemy 2 async engine & SessionLocal generator
│   │   ├── dependencies.py                  # FastAPI dependency: get_current_user via JWT cookie
│   │   ├── download_translation_model.py    # CLI setup script: downloads IndicTrans2 / M2M100
│   │   ├── llm.py                           # Ollama client: Llama 3.2 3B summarization & chatbot QA
│   │   ├── main.py                          # FastAPI entry point, routing, NDJSON streaming loops
│   │   ├── models.py                        # SQLAlchemy ORM models (5 schema tables)
│   │   ├── ocr.py                           # PyMuPDF block parser + OpenCV/PaddleOCR pipeline
│   │   ├── preflight.py                     # CLI diagnostic script: audits RAM, VRAM, DB, and models
│   │   ├── retrieval.py                     # Zero-VRAM in-memory BM25 chunk retrieval engine
│   │   ├── schemas.py                       # Pydantic request/response serialization models
│   │   ├── security.py                      # Bcrypt password hashing & PyJWT encode/decode routines
│   │   ├── summary.py                       # Extractive clinical summarizer & layperson expansions
│   │   ├── translation.py                   # IndicTrans2 / M2M100 loader, chunking & CUDA inference
│   │   └── worker.py                        # Background serialized translation queue worker
│   ├── migrations/
│   │   ├── env.py                           # Alembic async migration environment
│   │   └── versions/                        # Schema revisions (0001 -> 0002 -> 0003 -> 0004)
│   ├── tests/
│   │   ├── test_hardware_smoke.py           # Live GPU CUDA inference smoke test on GTX 1650
│   │   ├── test_queue_and_chat.py           # Isolation, queue ordering, and chat persistence tests
│   │   ├── test_security.py                 # JWT token round-trip, bcrypt & PBKDF2 tests
│   │   ├── test_summary.py                  # Extractive summary tests & regex rule assertions
│   │   └── test_translation.py              # IndicTrans2 tags, device selection & chunking tests
│   ├── uploads/                             # Local directory for isolated uploaded documents
│   ├── alembic.ini                          # Database migration configuration
│   ├── requirements.txt                     # Pinned backend dependencies
│   └── .env.example                         # Template for backend environment variables
├── frontend/
│   ├── app/
│   │   ├── globals.css                      # Custom design tokens, progress bars, responsive styles
│   │   ├── layout.tsx                       # Root HTML shell & metadata
│   │   └── page.tsx                         # Main Next.js React client (Dashboard, feeds, tabs, chat)
│   ├── lib/
│   │   └── api.ts                           # Typed fetch client, NDJSON stream reader, error classes
│   ├── package.json                         # Next.js 15, React 19, TypeScript dependencies
│   ├── tsconfig.json                        # Strict TypeScript compiler options
│   └── .env.local.example                   # Template for NEXT_PUBLIC_API_URL
├── demo/
│   ├── demo.tex                             # Comprehensive clinical H&P teaching vignette
│   └── demo.pdf                             # Compiled 2-page sample medical report for live demo
├── docker-compose.yml                       # PostgreSQL 16 container definition for local dev
├── run_medlingua.bat                        # Automated Windows 7-step hardware-aware orchestrator
├── MEDLINGUA_AUDIT_AND_NLP_GUIDE.md         # Comprehensive 30-section audit & academic defense guide
└── README.md                                # Project master documentation & user guide
```

---

## Configuration & Environment Variables

Backend settings are loaded from `backend/.env`. Frontend settings are loaded from `frontend/.env.local`.

| Variable | Component | Required | Default / Example | Description |
|---|---|:---:|---|---|
| `DATABASE_URL` | Backend | Yes | `postgresql+asyncpg://postgres:postgres@localhost:5432/medlingua` | Async SQLAlchemy PostgreSQL connection string. |
| `JWT_SECRET_KEY` | Backend | Yes | Replace with private random 48-char string | Secret key used to sign HS256 authentication tokens. |
| `JWT_EXPIRE_MINUTES` | Backend | No | `1440` (24 hours) | Token lifetime in minutes before requiring re-login. |
| `UPLOAD_DIR` | Backend | No | `./uploads` | Filesystem directory where uploaded files are stored. |
| `MAX_UPLOAD_SIZE_MB` | Backend | No | `20` | Maximum allowable upload file size in megabytes. |
| `MAX_PDF_PAGES` | Backend | No | `20` | Maximum page count accepted for PDF documents. |
| `FRONTEND_ORIGIN` | Backend | No | `http://localhost:3000` | Allowed origin for Cross-Origin Resource Sharing (CORS). |
| `HF_TOKEN` | Backend | No | `hf_xxxxxxxxxxxxxxxxxxxx` | Hugging Face Read-Access token for IndicTrans2. |
| `OLLAMA_BASE_URL` | Backend | No | `http://localhost:11434` | HTTP base URL for local Ollama service. |
| `OLLAMA_MODEL` | Backend | No | `llama3.2:3b` | Target LLM model tag in Ollama. |
| `NEXT_PUBLIC_API_URL` | Frontend | No | `http://localhost:8000` | Base API URL used by browser fetch requests. |

---

## Privacy, Security & Medical Disclaimer

### Patient Privacy Guarantees
- **100% Local Inference:** No clinical documents, extracted patient records, summaries, or translations are ever transmitted to third-party cloud APIs (such as OpenAI, Google Translate, or AWS Comprehend).
- **Offline LLM & NMT:** Summarization and translation models execute directly on your local GPU or CPU. The Hugging Face downloader only accesses the internet during one-time initial setup to fetch model weights.
- **Isolated Record Storage:** Every upload is isolated with a distinct UUID4; records cannot collide or overwrite each other.

### Application Security
- **Bcrypt Password Security:** Passwords are hashed with native bcrypt using individual cryptographic salts and safe 72-byte truncation. Backward-compatible verification ensures existing accounts using PBKDF2 remain accessible.
- **JWT Cookie Protection:** Authentication tokens are stored in `HttpOnly`, `SameSite=Lax` cookies, shielding sessions against Cross-Site Scripting (XSS) attacks. For HTTPS deployments, `secure=True` and `samesite="none"` are supported.
- **Strict Authorization:** All document lookups, translations, and chat histories are scoped to the authenticated `owner_id`. Users cannot view or query records belonging to other accounts.

### Medical & Legal Disclaimer
> [!IMPORTANT]
> **MedLingua is an informational and linguistic reading aid, NOT a clinical diagnostic tool or medical device.**
> - Summaries and translations generated by this software are provided strictly for educational and communication support.
> - Natural language processing and machine translation models can occasionally omit details, misinterpret complex clinical negations, or miscalculate numerical dosages.
> - **Always verify all clinical information against the original physical medical record and consult a licensed healthcare professional before making any treatment, dietary, or medication decisions.**

---

## Testing Strategy & Test Suite Results

MedLingua includes a rigorous testing suite covering unit logic, integration flows, and live hardware execution.

### Running Backend Tests
From the `backend` directory with `.venv` active:

```cmd
python -m pytest tests -v
```

**Verification Results: All 31 Tests Passing (100% Pass Rate):**
- `tests/test_security.py` (4 tests): JWT token generation, expiration checks, native bcrypt hashing, and backward-compatible PBKDF2 verification.
- `tests/test_summary.py` (13 tests): TF-IDF sentence salience scoring, heading segmentation, length normalization penalties, section budget allocation, and regex abbreviation expansions.
- `tests/test_translation.py` (6 tests): IndicTrans2 language tags (`eng_Latn`, `hin_Deva`, `mar_Deva`, `tam_Taml`), CUDA/CPU device resolution, subword chunking boundary preservation, and offline fallback mechanics.
- `tests/test_queue_and_chat.py` (5 tests): Upload record isolation (UUID4 uniqueness), strict sequential translation queue ordering, click-order FIFO re-translation prioritization, and multi-turn chat persistence.
- `tests/test_hardware_smoke.py` (3 tests): Live smoke test verifying real FP16 inference on the host's NVIDIA GeForce GTX 1650 with AI4Bharat IndicTrans2.

### Running Frontend Typecheck & Build
From the `frontend` directory:

```cmd
npm run lint
npm run build
```
- `tsc --noEmit`: **0 TypeScript errors.**
- `npm run build`: Production Next.js 15.5 bundle compiled successfully.

---

## Troubleshooting & Compatibility Guide

| Problem / Error Message | Root Cause | Verified Resolution |
|---|---|---|
| `ModuleNotFoundError: No module named 'transformers.onnx'` | AI4Bharat's `configuration_indictrans.py` imports deprecated `transformers.onnx` removed in Transformers $\ge 4.40$. | Handled automatically by MedLingua's runtime in-memory module shim in `backend/app/translation.py`. |
| `AttributeError: IndicTransTokenizer has no attribute _special_tokens_map` | `IndicTransTokenizer.__init__` assigns special tokens before calling `super().__init__()` in Transformers 5.x. | Resolved automatically by MedLingua's `PreTrainedTokenizerBase.__new__` pre-initialization monkey-patch. |
| `TypeError: tie_weights() got an unexpected keyword argument 'recompute_mapping'` | Modern Transformers calls `tie_weights(recompute_mapping=False)`, not supported by IndicTrans2. | Resolved by monkey-patching `tie_weights` in `transformers.dynamic_module_utils` and `auto_factory`. |
| `TypeError: 'EncoderDecoderCache' object is not subscriptable` | Transformers 5.x returns an `EncoderDecoderCache` object instead of a tuple for `past_key_values`. | Resolved by passing `use_cache=False` to `model.generate()`. |
| `NameError: name 're' is not defined` in `llm.py` | Missing `import re` statement for preamble cleaning regex. | Fixed: `import re` restored to top-level imports in `backend/app/llm.py`. |
| `401 Client Error: Cannot access gated repo` on IndicTrans2 | Missing Hugging Face token or license agreement on `ai4bharat/indictrans2-en-indic-dist-200M`. | 1. Accept license at [Hugging Face](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M).<br>2. Add `HF_TOKEN=hf_...` (Read access) to `backend/.env`.<br>3. Or rely on automatic Meta M2M100 fallback. |
| Hugging Face cache only contains ~15 KB of files | Configuration loading failed before weight binary (`model.safetensors`, 1.05 GB) was downloaded. | Run `python -m app.download_translation_model --model indictrans2` to trigger complete download. |
| Ollama connection error / `ConnectionRefusedError` | Ollama service is stopped or not listening on port 11434. | Start Ollama via `ollama serve` and verify with `curl http://localhost:11434/api/tags`. |
| `CUDA out of memory` during translation | Another GPU process is running or large batch exceeds 4 GB VRAM. | MedLingua automatically clears cache and falls back to CPU. Ensure `keep_alive: 0` is set for Ollama. |
| `alembic current head` reports unrecognized argument | `current` and `head` are separate commands. | Run `alembic current` to inspect applied revision; run `alembic upgrade head` to apply migrations. |
| Frontend displays "Cannot connect to backend" | FastAPI is stopped or `NEXT_PUBLIC_API_URL` is misconfigured. | Verify backend is running on `http://localhost:8000` and check `http://localhost:8000/health`. |

---

## Deprecations, Replacements & Architecture Evolution

This section documents all systems, models, and workflows that were updated, replaced, or deprecated during development:

| Subsystem | Previous Legacy Implementation | Current Modern Implementation | Rationale & Architectural Justification |
|---|---|---|---|
| **Primary NMT Model** | Meta M2M100-418M (`facebook/m2m100_418M`) | **AI4Bharat IndicTrans2-dist-200M** (`ai4bharat/indictrans2-en-indic-dist-200M`) | IndicTrans2 delivers state-of-the-art translation accuracy specifically tailored for Indian vernaculars. M2M100 is retained as an offline fallback. |
| **Model Weight Format** | Legacy `pytorch_model.bin` (Pickle format) | **Memory-Mapped `model.safetensors` (1,047.54 MB)** | SafeTensors prevents arbitrary code execution risks and enables zero-copy loading directly into GPU memory, saving 1.1 GB disk space. |
| **Clinical Summarization** | Pure rule-based extractive ranking only | **Dual Engine: Llama 3.2 3B Instruct (Ollama) + Extractive Fallback** | Llama 3.2 3B produces fluent, highly structured layperson summaries. Extractive scoring is retained as an automatic fallback if Ollama is offline. |
| **Password Security** | Passlib PBKDF2-SHA256 | **Native Bcrypt (`bcrypt.hashpw`) with PBKDF2 compatibility** | Modernizes cryptography, eliminates Passlib deprecation warnings, and retains full backward compatibility for existing user accounts. |
| **Upload Progress** | Static loading spinner (`"Processing..."`) | **Real-Time NDJSON Progress Feed (`UploadProgressFeed`)** | Provides granular visual stages (`saving_file` $\to$ `extracting_ocr` $\to$ `generating_summary` $\to$ `completed`). |
| **Translation Execution** | Uncoordinated ad-hoc translation | **Deterministic Sequential Pipeline + FIFO Click Queue** | Eliminates GPU memory thrashing on 4 GB VRAM. Translates Hindi $\to$ Marathi $\to$ Tamil sequentially, with cross-tab queue transparency. |
| **Document QA** | Not implemented (static document viewing only) | **Document-Grounded Chatbot with In-Memory BM25** | Enables interactive conversational queries strictly grounded in document text with zero VRAM overhead ($<1$ MB RAM). |
| **Storage Handling** | Duplicate uploads overwritten / collapsed | **UUID4 Record Isolation & Unique Disk Storage** | Prevents race conditions and ensures distinct conversation histories and translation queues for every upload. |

---

## Academic Viva Voce Defense, Demonstration & NLP Glossary

### 1. Two-Minute Elevator Pitch
> *"Good morning, respected examiners. My project is **MedLingua**, a private, local-first medical natural language processing platform that transforms complex clinical reports into plain-language summaries, vernacular translations in Hindi, Marathi, and Tamil, and conversational document QA on consumer hardware.*
>
> *In clinical healthcare, patients frequently receive medical records filled with dense, acronym-heavy English that they cannot understand. Existing online translators send sensitive patient records to third-party cloud APIs, violating medical privacy.*
>
> *MedLingua addresses both issues locally on a consumer NVIDIA GTX 1650 GPU:*
> 1. *It ingests vector and scanned documents using a hybrid **PyMuPDF and PaddleOCR** pipeline with adaptive image preprocessing.*
> 2. *It generates a structured, four-part plain-language summary using local **Llama 3.2 3B Instruct** with an automatic **extractive TF-IDF fallback**, expanding medical acronyms into everyday language.*
> 3. *It translates both the summary and source report using **AI4Bharat IndicTrans2**, the state-of-the-art neural model for Indian languages, executed sequentially to prevent GPU memory thrashing.*
> 4. *It provides a **grounded clinical chatbot** using zero-VRAM BM25 retrieval, streaming answers with strict anti-hallucination guardrails.*
>
> *This demonstrates a complete end-to-end application of layout analysis, subword tokenization, domain-adapted scoring, neural sequence-to-sequence translation, and memory-constrained inference optimization."*

---

### 2. Comprehensive NLP Glossary

- **BPE (Byte-Pair Encoding):** A subword tokenization algorithm that iteratively merges the most frequent pairs of characters or bytes, preventing Out-Of-Vocabulary (OOV) errors on rare medical terminology.
- **Encoder-Decoder Architecture:** A sequence-to-sequence model design where an encoder processes the source sequence into continuous contextual vectors, and an autoregressive decoder generates target tokens while attending to encoder states.
- **Self-Attention vs. Cross-Attention:** Self-attention computes dependencies between all tokens within the same sequence. Cross-attention allows decoder target tokens (e.g., Hindi) to attend to source tokens (English) from the encoder.
- **Greedy Decoding vs. Beam Search:** Greedy decoding selects the highest-probability token at each step ($B=1$), minimizing latency and VRAM use. Beam search tracks the top $B$ paths, improving fluency at the expense of higher computation.
- **Repetition Penalty:** An inference penalty applied to the logits of previously generated tokens, preventing repetitive autoregressive loops.
- **Extractive vs. Abstractive Summarization:** Extractive summarization selects verbatim sentences directly from the source text. Abstractive summarization synthesizes novel sentences using a generative decoder.
- **BLEU & chrF++:** BLEU measures n-gram precision with brevity penalty. chrF++ measures character n-gram F-score, which is significantly more accurate for morphologically rich Indian languages.
- **ROUGE:** Recall-oriented metric measuring unigram (ROUGE-1), bigram (ROUGE-2), and longest common subsequence (ROUGE-L) overlap.
- **Okapi BM25:** A probabilistic ranking algorithm calculating document relevance based on term frequency and document length normalization.
- **NDJSON (Newline-Delimited JSON):** A streaming protocol where individual JSON objects are separated by newlines (`\n`), enabling incremental client rendering over a single HTTP connection.

---

### 3. Top 25 Viva Voce Questions & Defensible Answers

#### Fundamentals & Project Scope
1. **Q: What primary problem does MedLingua solve?**
   - **A:** It bridges the clinical comprehension and language barrier for patients by extracting, summarizing, and translating dense English medical reports into plain language and Indian vernaculars (Hindi, Marathi, Tamil) on private local hardware.
2. **Q: Why didn't you use cloud LLM APIs like OpenAI ChatGPT or Google Cloud Translation?**
   - **A:** Two fundamental reasons: **Patient Privacy** and **Cost/Offline Accessibility**. Medical reports contain Protected Health Information (PHI). Sending them to third-party APIs compromises privacy and violates healthcare compliance standards. MedLingua executes 100% locally with zero cloud telemetry.
3. **Q: What is the core technical contribution of your project?**
   - **A:** Designing a hardware-optimized, end-to-end clinical NLP architecture on a consumer 4 GB GPU that integrates vector/OCR extraction, instruction-tuned summarization with extractive fallback, SOTA Indic neural machine translation, and a zero-VRAM grounded chatbot.

#### NLP Architecture & Summarization
4. **Q: How does the summarization pipeline guarantee factual accuracy?**
   - **A:** MedLingua uses a dual-tier strategy: Llama 3.2 3B is prompt-engineered with low temperature ($0.2$) and strict provenance constraints. If Ollama is offline or generates hallucinated content, the system falls back to a deterministic extractive TF-IDF algorithm that lifts sentences verbatim from the report.
5. **Q: What is the mathematical formulation of your extractive scoring algorithm?**
   - **A:** It scores sentences using term frequency multiplied by inverse document frequency, normalized by the square root of sentence length, plus additive domain bonuses: $+2$ for clinical keywords, $+6$ for pathology findings, and $+1.5$ for lab measurement units.
6. **Q: How does MedLingua handle document layout and clinical sections?**
   - **A:** It evaluates 37 regular expression patterns across five canonical clinical sections (HPI, History, Findings, Assessment, Plan), allocating proportional word budgets to each section.
7. **Q: How are medical abbreviations made understandable to laypersons?**
   - **A:** A regex lexical substitution dictionary expands 42+ common acronyms (e.g., `SOB` $\to$ `shortness of breath (SOB)`, `HTN` $\to$ `high blood pressure (hypertension)`).

#### Translation & Sequence-to-Sequence Modeling
8. **Q: Which translation model is used, and why is it superior to general-domain models?**
   - **A:** AI4Bharat **IndicTrans2-dist-200M**, designed specifically for 22 Indian languages by IIT Madras. It outperforms general-domain models (like Google Translate or M2M100) on Indic benchmarks due to specialized script normalization and subword vocabularies.
9. **Q: Why does MedLingua also retain Meta M2M100?**
   - **A:** As an offline, ungated fallback. IndicTrans2 requires accepting an academic license on Hugging Face; if a user has not configured `HF_TOKEN`, the system falls back to M2M100 without crashing.
10. **Q: How does MedLingua handle long reports that exceed Transformer context limits?**
    - **A:** It implements a two-tier boundary-preserving chunker that splits on paragraphs and sentences, grouping tokens into 256-subword windows so attention complexity remains manageable and no sentence is truncated mid-word.
11. **Q: Why did you choose greedy decoding over beam search for translation?**
    - **A:** To minimize inference latency and VRAM usage on the host GTX 1650 (4 GB). Greedy decoding ($B=1$) executes in half the time and consumes $\approx 4\times$ less VRAM than beam search ($B=4$), while `no_repeat_ngram_size=3` prevents repetition loops.
12. **Q: What was the ONNX import issue on IndicTrans2 and how did you resolve it?**
    - **A:** The model's 2023 configuration file executed `from transformers.onnx import OnnxConfig`, which was deleted in modern Transformers. We resolved it by dynamically injecting mock module classes into `sys.modules["transformers.onnx"]`.
13. **Q: How did you resolve the `_special_tokens_map` AttributeError in Transformers 5.x?**
    - **A:** AI4Bharat's tokenizer assigned special token attributes before executing `super().__init__()`. We wrapped `PreTrainedTokenizerBase.__new__` to initialize `_special_tokens_map` prior to attribute assignment.
14. **Q: Why did you prioritize `model.safetensors` over `pytorch_model.bin`?**
    - **A:** SafeTensors uses zero-copy memory mapping directly into GPU memory, loads faster, and avoids the arbitrary code execution vulnerabilities of Python `pickle`, while saving 1.1 GB of duplicate storage.

#### Systems, Memory & Hardware Optimization
15. **Q: How do you prevent CUDA Out-Of-Memory errors on a 4 GB GTX 1650?**
    - **A:** Through **VRAM Serialization**: a process-wide `_MODEL_LOCK` ensures Ollama and PyTorch never infer simultaneously. Ollama is called with `keep_alive: 0` to unload the LLM immediately after summarization, freeing 2.0 GB VRAM before IndicTrans2 (1.05 GB) loads. OCR runs strictly on CPU.
16. **Q: How does the sequential multi-language translation pipeline work?**
    - **A:** Document upload initiates a deterministic chain: Hindi summary $\to$ Hindi chunks $\to$ Marathi summary $\to$ Marathi chunks $\to$ Tamil summary $\to$ Tamil chunks. Re-translate buttons remain disabled until all three complete.
17. **Q: How does cross-tab queue transparency work in the UI?**
    - **A:** While Hindi translates, the Marathi and Tamil tabs render `QueuedProgressFeed` showing their position in queue and active status, preventing confusing "loading" or "initiating LLM" notices.
18. **Q: How does click-order FIFO re-translation work?**
    - **A:** When a user requests re-translation, languages are processed in the chronological order clicked, with per-button disabled states and dynamic position badges.
19. **Q: What happens if a translation job is interrupted halfway through?**
    - **A:** Completed chunks are persisted incrementally to PostgreSQL JSONB. When restarted, the engine loads finished chunks from the database and resumes inference from the first uncompleted chunk.
20. **Q: How does the grounded chatbot work without consuming GPU VRAM?**
    - **A:** It uses an in-memory Okapi BM25 index ($k_1=1.5, b=0.75$) that retrieves top-3 relevant 250-word chunks in $<2$ ms using $<1$ MB of RAM and 0 MB VRAM, passing them as grounded context to Llama 3.2 3B.

#### Security, Evaluation & Defense
21. **Q: How is user authentication secured?**
    - **A:** Passwords are encrypted with native bcrypt (with PBKDF2 backward compatibility). Authentication uses HS256 JWT tokens stored in `HttpOnly`, `SameSite=Lax` cookies to prevent XSS attacks.
22. **Q: Can MedLingua run on systems without an NVIDIA GPU?**
    - **A:** Yes. PyTorch dynamically checks `torch.cuda.is_available()`. If CUDA is absent, models execute seamlessly on CPU.
23. **Q: How would you evaluate the translation quality scientifically?**
    - **A:** Using **chrF++** and **BLEU** against human reference translations produced by bilingual medical experts. chrF++ is prioritized because character n-grams handle Indic morphology far better than word-level BLEU.
24. **Q: How would you evaluate the summarization pipeline scientifically?**
    - **A:** Using **ROUGE-1, ROUGE-2, and ROUGE-L** against expert clinical reference summaries, alongside **FactCC** to evaluate factual entailment and absence of hallucinations.
25. **Q: Why is MedLingua suitable for a final-year Computer Science / NLP degree?**
    - **A:** It demonstrates mastery across the full modern NLP lifecycle: computer vision document layout analysis (OCR), subword tokenization (BPE), domain-adapted algorithmic information extraction (TF-IDF), neural sequence-to-sequence generation (Transformers), prompt engineering, memory-constrained hardware orchestration, and full-stack reactive streaming.

---

### 4. Step-by-Step Examiner Demonstration Checklist

```text
Step 1: System Preflight & Hardware Inspection
- Action: Open terminal in backend and run "python -m app.preflight".
- Explain: Show that RAM, GTX 1650 CUDA VRAM, PostgreSQL schema, Ollama service, and translation safetensors are all verified green.

Step 2: Authenticated Workspace & Document Upload
- Action: Navigate to http://localhost:3000 and log in.
- Action: Upload "demo/demo.pdf" (the clinical H&P teaching vignette).
- Point Out: Show the multi-stage UploadProgressFeed (Saving File -> OCR Extraction -> Clinical Summarization -> Complete).

Step 3: Grounded Plain-Language Summary
- Action: Review Section 01 "The Essentials (Summary)".
- Point Out:
  - Four structured paragraphs: Symptoms (HPI), Medical History, Clinical Findings, Diagnosis & Plan.
  - Acronym expansions: "SOB" -> "shortness of breath (SOB)", "HTN" -> "high blood pressure (hypertension)".
  - Provenance badge: "LLM Generated" or "Extractive Fallback".

Step 4: Deterministic Sequential Translation
- Action: Direct examiner's attention to language tabs.
- Point Out:
  - Hindi Tab: TranslationProgressFeed shows active section progress (X/N sections), animated progress bar, and live translated snippets.
  - Marathi & Tamil Tabs: QueuedProgressFeed shows "Position #1 in Queue" and "Hindi translation in progress".
  - Re-translate buttons are disabled during pipeline execution.

Step 5: Resumption & Database Persistence
- Action: Refresh browser during translation. Reopen document.
- Point Out: Completed chunks load instantly from PostgreSQL JSONB; inference resumes from the next chunk without re-running finished text.

Step 6: Grounded Document Chatbot
- Action: Open the chatbot sidebar and ask: "What dosage of Metformin was prescribed?"
- Point Out: Tokens stream in real time via SSE; answer is strictly grounded in the document text with clinical provenance.

Step 7: Clinician Comparison Verification
- Action: Switch between English and Hindi tabs.
- Point Out: Original English extracted text remains side-by-side with vernacular translations so clinicians can verify findings against the source.
```

---

### 5. Recommended Academic Project Report Outline

```text
Chapter 1: Introduction & Problem Formulation
  1.1 Clinical Communication Gaps in Vernacular Healthcare
  1.2 Challenges in Medical Report Comprehension & Acronym Density
  1.3 Privacy Risks of Third-Party Cloud AI in Healthcare
  1.4 Project Objectives & Scope of MedLingua

Chapter 2: Literature Review & Related Work
  2.1 Document Layout Analysis & Medical OCR (PyMuPDF & PaddleOCR)
  2.2 Extractive vs. Abstractive Summarization in Clinical Informatics
  2.3 Multilingual Neural Machine Translation (M2M100 & AI4Bharat IndicTrans2)
  2.4 Grounded Question Answering & In-Memory Information Retrieval (BM25)

Chapter 3: System Architecture & Hardware Optimization
  3.1 Full-Stack Client-Server Architecture (Next.js 15 & FastAPI)
  3.2 Hardware Profile & Constraints (NVIDIA GeForce GTX 1650 4 GB)
  3.3 VRAM Serialization Coordinator & Ollama Memory Management
  3.4 Database Schema & PostgreSQL JSONB Resumption Cache

Chapter 4: Natural Language Processing & Machine Learning Pipelines
  4.1 Hybrid Document Ingestion & Image Preprocessing Pipeline
  4.2 Clinical Summarization Engine: Llama 3.2 3B & TF-IDF Extractive Fallback
  4.3 Plain-Language Lexical Expansion Engine
  4.4 Subword Chunking & Token Window Management
  4.5 AI4Bharat IndicTrans2 Neural Translation & Transformers 5.x Shims
  4.6 Document-Grounded Chatbot & Okapi BM25 Retrieval Formulation

Chapter 5: Implementation, Diagnostics & Experimental Verification
  5.1 Hardware-Aware Automated Toolchain (run_medlingua.bat & preflight.py)
  5.2 Multi-Stage Streaming Protocol (NDJSON & SSE)
  5.3 Empirical Benchmark Results (Qwen vs. Llama 3.2 3B)
  5.4 Unit, Integration & Live Hardware Test Suite Results

Chapter 6: Conclusion, Limitations & Future Enhancements
  6.1 Summary of Contributions
  6.2 Limitations & Clinical Safety Considerations
  6.3 Future Work: BioBERT Named Entity Recognition & Quantized CTranslate2
```

---

## Contributing

1. Fork the repository and create a feature branch (`git checkout -b feature/clinical-ner`).
2. Adhere to project architecture: ensure all inferences remain strictly local and private.
3. Add tests to `backend/tests/` covering new endpoints or NLP logic.
4. Verify backend tests pass (`python -m pytest tests -v`) and frontend builds without error (`npm run build`).
5. Submit a pull request detailing the changes, validation performed, and setup implications.

Please do not commit real patient records, API keys, virtual environments, or downloaded model weights to Git.

---

## License

This project is licensed under the **MIT License**.

---

Made with ❤️ by [Tanishq Mudaliar](https://github.com/tanishqmudaliar)

*Understand the report clearly. Always verify against the original.*
