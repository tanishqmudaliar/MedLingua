from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import json
import logging
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from sqlalchemy import delete, desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .config import get_settings
from .db import get_db, SessionLocal
from .dependencies import get_current_user
from .llm import check_gpu_diagnostics, generate_document_summary, stream_chatbot_response
from .models import ChatMessage, Conversation, Document, TranslationJob, User
from .ocr import extract_image_text, extract_pdf_text
from .schemas import (
    ChatMessageRead,
    ChatRequest,
    ConversationRead,
    DocumentRead,
    DocumentTranslationRequest,
    RetranslateRequest,
    UserCreate,
    UserLogin,
    UserRead,
)
from .security import create_access_token, hash_password, verify_password
from .translation import TranslationModelUnavailable, translate_summary, translate_text
from .worker import translation_worker_loop

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background serialized translation worker loop
    worker_task = asyncio.create_task(translation_worker_loop())
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="MedLingua API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"}


def _set_auth_cookie(response: Response, user_id: UUID) -> None:
    is_prod = settings.frontend_origin.startswith("https://")
    response.set_cookie(
        key="access_token",
        value=create_access_token(user_id),
        httponly=True,
        samesite="none" if is_prod else "lax",
        secure=is_prod,
        max_age=settings.jwt_expire_minutes * 60,
    )


def _clear_auth_cookie(response: Response) -> None:
    is_prod = settings.frontend_origin.startswith("https://")
    response.delete_cookie(
        key="access_token",
        httponly=True,
        samesite="none" if is_prod else "lax",
        secure=is_prod,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/system/hardware")
async def hardware_status(user: User = Depends(get_current_user)) -> dict[str, object]:
    """Returns runtime GPU VRAM, active Ollama models, and layer offload diagnostics."""
    return check_gpu_diagnostics()


@app.post("/auth/signup", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def signup(payload: UserCreate, response: Response, db: AsyncSession = Depends(get_db)) -> User:
    existing = await db.scalar(
        select(User).where(or_(User.username == payload.username, User.email == payload.email.lower()))
    )
    if existing:
        raise HTTPException(status_code=409, detail="Username or email is already registered")
    user = User(
        username=payload.username,
        name=payload.name,
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    _set_auth_cookie(response, user.id)
    return user


@app.post("/auth/login", response_model=UserRead)
async def login(payload: UserLogin, response: Response, db: AsyncSession = Depends(get_db)) -> User:
    user = await db.scalar(select(User).where(User.username == payload.username))
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found. Sign up to create your account.")
    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect password")
    _set_auth_cookie(response, user.id)
    return user


@app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> None:
    _clear_auth_cookie(response)


@app.get("/auth/me", response_model=UserRead)
async def me(user: User = Depends(get_current_user)) -> User:
    return user


@app.post("/documents", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Every upload creates a new independent record with its own unique UUID and storage path.
    Supports real-time NDJSON progress streaming when Accept: application/x-ndjson is present.
    """
    if file.content_type != "application/pdf" and file.content_type not in IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Only PDF, PNG, JPG, JPEG, WEBP, BMP, or TIFF files are supported")
    data = await file.read(settings.max_upload_size_bytes + 1)
    if len(data) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds the {settings.max_upload_size_mb} MB limit")

    is_stream = "application/x-ndjson" in request.headers.get("accept", "")

    if is_stream:
        async def upload_event_stream():
            queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def emit(evt: dict[str, object]) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, evt)

            async def process_upload():
                stored_path = None
                try:
                    emit({
                        "type": "stage",
                        "stage": "uploading",
                        "message": "Validating file format and persisting upload...",
                        "percent": 20,
                    })
                    suffix = Path(file.filename or "upload").suffix.lower() or ".bin"
                    settings.upload_dir.mkdir(parents=True, exist_ok=True)
                    stored_path = settings.upload_dir / f"{user.id}_{uuid4().hex}{suffix}"
                    stored_path.write_bytes(data)

                    emit({
                        "type": "stage",
                        "stage": "extracting",
                        "message": "Extracting text with PyMuPDF / PaddleOCR...",
                        "percent": 45,
                    })
                    text = await run_in_threadpool(
                        extract_pdf_text, stored_path, settings.max_pdf_pages
                    ) if file.content_type == "application/pdf" else await run_in_threadpool(extract_image_text, data)

                    emit({
                        "type": "stage",
                        "stage": "summarizing",
                        "message": "Analyzing clinical report with Llama 3.2 3B...",
                        "percent": 75,
                    })
                    summary, summary_method = await run_in_threadpool(generate_document_summary, text)

                    emit({
                        "type": "stage",
                        "stage": "saving",
                        "message": "Structuring clinical sections and saving to database...",
                        "percent": 90,
                    })
                    document = Document(
                        id=uuid4(),
                        owner_id=user.id,
                        original_filename=file.filename or "upload",
                        media_type=file.content_type,
                        storage_path=str(stored_path),
                        extracted_text=text,
                        summary=summary or None,
                        summary_method=summary_method if summary else None,
                        processing_status="translating" if summary else "completed",
                        translations_status={"hi": "pending", "mr": "pending", "ta": "pending"} if summary else {},
                    )
                    db.add(document)

                    conversation = Conversation(
                        id=uuid4(),
                        document_id=document.id,
                        user_id=user.id,
                    )
                    db.add(conversation)

                    if summary:
                        jobs = [
                            TranslationJob(id=uuid4(), document_id=document.id, language="hi", order_index=0, status="pending"),
                            TranslationJob(id=uuid4(), document_id=document.id, language="mr", order_index=1, status="pending"),
                            TranslationJob(id=uuid4(), document_id=document.id, language="ta", order_index=2, status="pending"),
                        ]
                        db.add_all(jobs)

                    await db.commit()
                    await db.refresh(document)

                    doc_data = DocumentRead.model_validate(document).model_dump(mode="json")
                    emit({
                        "type": "complete",
                        "stage": "completed",
                        "message": "Clinical summary ready!",
                        "document": doc_data,
                        "percent": 100,
                    })
                except Exception as exc:
                    if stored_path and stored_path.exists():
                        stored_path.unlink(missing_ok=True)
                    logger.exception("Streaming upload processing failed: %s", exc)
                    emit({
                        "type": "error",
                        "stage": "failed",
                        "message": str(exc),
                        "status": 500,
                    })
                finally:
                    emit({"type": "end"})

            task = asyncio.create_task(process_upload())
            try:
                while True:
                    evt = await queue.get()
                    if evt.get("type") == "end":
                        break
                    yield json.dumps(evt, ensure_ascii=False) + "\n"
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(upload_event_stream(), media_type="application/x-ndjson")

    # Standard non-streaming branch
    suffix = Path(file.filename or "upload").suffix.lower() or ".bin"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = settings.upload_dir / f"{user.id}_{uuid4().hex}{suffix}"
    stored_path.write_bytes(data)

    try:
        text = (
            extract_pdf_text(stored_path, settings.max_pdf_pages)
            if file.content_type == "application/pdf"
            else extract_image_text(data)
        )
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        stored_path.unlink(missing_ok=True)
        if isinstance(exc, (ImportError, OSError)):
            detail = "PaddleOCR could not load. Check the pinned PaddleOCR/PaddlePaddle dependencies."
            raise HTTPException(status_code=503, detail=detail) from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Generate English summary using local LLM with extractive fallback
    summary, summary_method = await run_in_threadpool(generate_document_summary, text)

    # Create new independent document record
    document = Document(
        id=uuid4(),
        owner_id=user.id,
        original_filename=file.filename or "upload",
        media_type=file.content_type,
        storage_path=str(stored_path),
        extracted_text=text,
        summary=summary or None,
        summary_method=summary_method if summary else None,
        processing_status="translating" if summary else "completed",
        translations_status={"hi": "pending", "mr": "pending", "ta": "pending"} if summary else {},
    )
    db.add(document)

    # Initialize independent persistent conversation for this document
    conversation = Conversation(
        id=uuid4(),
        document_id=document.id,
        user_id=user.id,
    )
    db.add(conversation)

    # Only after extraction and English summarization succeed, schedule initial translations in strict order:
    # Hindi (0) -> Marathi (1) -> Tamil (2)
    if summary:
        jobs = [
            TranslationJob(id=uuid4(), document_id=document.id, language="hi", order_index=0, status="pending"),
            TranslationJob(id=uuid4(), document_id=document.id, language="mr", order_index=1, status="pending"),
            TranslationJob(id=uuid4(), document_id=document.id, language="ta", order_index=2, status="pending"),
        ]
        db.add_all(jobs)

    await db.commit()
    await db.refresh(document)
    return document


@app.get("/documents", response_model=list[DocumentRead])
async def list_documents(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> list[Document]:
    result = await db.scalars(select(Document).where(Document.owner_id == user.id).order_by(desc(Document.created_at)))
    return list(result)


@app.get("/documents/{document_id}", response_model=DocumentRead)
async def get_document(document_id: UUID, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> Document:
    document = await db.scalar(select(Document).where(Document.id == document_id, Document.owner_id == user.id))
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@app.post("/documents/{document_id}/retranslate", response_model=DocumentRead)
async def retranslate_document(
    document_id: UUID,
    payload: RetranslateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Document:
    """Manually re-runs translation for ONLY the explicitly selected language."""
    document = await db.scalar(select(Document).where(Document.id == document_id, Document.owner_id == user.id))
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not document.summary:
        raise HTTPException(status_code=400, detail="Document has no English summary to translate")

    lang = payload.language
    # Update translation status for this specific language
    t_status = dict(document.translations_status or {})
    t_status[lang] = "pending"
    document.translations_status = t_status
    document.processing_status = "translating"

    # Enqueue a dedicated manual translation job
    manual_job = TranslationJob(
        id=uuid4(),
        document_id=document.id,
        language=lang,
        order_index=0,
        status="pending",
        is_manual=True,
    )
    db.add(manual_job)
    await db.commit()
    await db.refresh(document)
    return document


@app.get("/documents/{document_id}/conversation", response_model=ConversationRead)
async def get_conversation(
    document_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Conversation:
    """Retrieves the persistent conversation and all message history for a record."""
    document = await db.scalar(select(Document).where(Document.id == document_id, Document.owner_id == user.id))
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    conversation = await db.scalar(
        select(Conversation)
        .where(Conversation.document_id == document_id, Conversation.user_id == user.id)
        .options(selectinload(Conversation.messages))
    )
    if not conversation:
        conversation = Conversation(id=uuid4(), document_id=document.id, user_id=user.id)
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation, ["messages"])

    return conversation


@app.post("/documents/{document_id}/chat")
async def chat_with_document(
    document_id: UUID,
    payload: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Streams chatbot response to question grounded in document content and summaries."""
    document = await db.scalar(select(Document).where(Document.id == document_id, Document.owner_id == user.id))
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    conversation = await db.scalar(
        select(Conversation)
        .where(Conversation.document_id == document_id, Conversation.user_id == user.id)
        .options(selectinload(Conversation.messages))
    )
    if not conversation:
        conversation = Conversation(id=uuid4(), document_id=document.id, user_id=user.id)
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation, ["messages"])

    # Persist the user message immediately
    user_msg = ChatMessage(
        id=uuid4(),
        conversation_id=conversation.id,
        role="user",
        content=payload.message.strip(),
    )
    db.add(user_msg)
    await db.commit()

    # Prepare chat history and metadata
    history = [{"role": m.role, "content": m.content} for m in conversation.messages]
    vernacular_summaries = {
        "hi": document.hindi_summary,
        "mr": document.marathi_summary,
        "ta": document.tamil_summary,
    }
    metadata = {
        "filename": document.original_filename,
        "created_at": str(document.created_at),
    }

    async def event_generator():
        accumulated_tokens: list[str] = []
        try:
            async for token in stream_chatbot_response(
                extracted_text=document.extracted_text,
                english_summary=document.summary,
                vernacular_summaries=vernacular_summaries,
                metadata=metadata,
                history=history,
                user_query=payload.message.strip(),
            ):
                accumulated_tokens.append(token)
                yield json.dumps({"type": "token", "content": token}, ensure_ascii=False) + "\n"

            assistant_reply = "".join(accumulated_tokens).strip()
            if not assistant_reply:
                assistant_reply = "This document does not contain enough information to answer that question."

            # Save assistant message in PostgreSQL
            async with SessionLocal() as save_db:
                assistant_msg = ChatMessage(
                    id=uuid4(),
                    conversation_id=conversation.id,
                    role="assistant",
                    content=assistant_reply,
                )
                save_db.add(assistant_msg)
                await save_db.commit()

            yield json.dumps({"type": "complete", "content": assistant_reply}, ensure_ascii=False) + "\n"
        except Exception as exc:
            logger.exception("Chatbot streaming error: %s", exc)
            yield json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/documents/{document_id}/translate")
async def translate_document(
    document_id: UUID,
    payload: DocumentTranslationRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Backward-compatible streaming endpoint for real-time translation."""
    document = await db.scalar(
        select(Document).where(Document.id == document_id, Document.owner_id == user.id)
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    async def events():
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        language = payload.language
        cached = {} if payload.force else dict(document.translations.get(language, {}))
        cached_summary = None if payload.force else cached.get("summary")
        cached_text = None if payload.force else cached.get("extracted_text")
        cached_chunks = [] if payload.force else list(cached.get("extracted_chunks", []))

        def emit(event: dict[str, object]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def translate_in_background() -> None:
            try:
                if cached_summary is None:
                    emit({"type": "status", "stage": "model", "message": "Loading translation engine (IndicTrans2 / M2M100)..."})
                    summary = translate_summary(document.summary or "", language)
                else:
                    summary = cached_summary
                emit({"type": "summary", "text": summary})

                if cached_text is not None:
                    emit({"type": "complete", "extracted_text": cached_text})
                    return

                def report_progress(completed: int, total: int, chunk: str) -> None:
                    emit({"type": "text_progress", "completed": completed, "total": total, "chunk": chunk})

                extracted_text = translate_text(
                    document.extracted_text,
                    language,
                    on_progress=report_progress,
                    cached_chunks=cached_chunks,
                )
                emit({"type": "complete", "extracted_text": extracted_text})
            except TranslationModelUnavailable as exc:
                emit({"type": "error", "message": str(exc), "status": 503})
            except Exception as exc:
                logger.exception("Translation failed: %s", exc)
                emit({"type": "error", "message": f"Translation failed: {exc}", "status": 500})
            finally:
                emit({"type": "end"})

        worker = asyncio.create_task(run_in_threadpool(translate_in_background))
        try:
            while True:
                event = await queue.get()
                if event["type"] == "end":
                    break
                current = dict(document.translations.get(language, {}))
                if event["type"] == "summary":
                    current["summary"] = event["text"]
                    if language == "hi":
                        document.hindi_summary = str(event["text"])
                    elif language == "mr":
                        document.marathi_summary = str(event["text"])
                    elif language == "ta":
                        document.tamil_summary = str(event["text"])
                elif event["type"] == "text_progress" and event["chunk"]:
                    chunks = list(current.get("extracted_chunks", []))
                    completed = int(event["completed"])
                    if len(chunks) < completed:
                        chunks.append(str(event["chunk"]))
                    current["extracted_chunks"] = chunks
                elif event["type"] == "complete":
                    current["extracted_text"] = event["extracted_text"]
                    current["extracted_chunks"] = []
                    t_status = dict(document.translations_status or {})
                    t_status[language] = "completed"
                    document.translations_status = t_status
                    if all(t_status.get(l) == "completed" for l in ("hi", "mr", "ta")):
                        document.processing_status = "completed"

                should_persist = (
                    event["type"] in {"summary", "complete"}
                    or (event["type"] == "text_progress" and bool(event["chunk"]))
                )
                if should_persist:
                    translations = dict(document.translations)
                    translations[language] = current
                    document.translations = translations
                    await db.commit()
                yield json.dumps(event, ensure_ascii=False) + "\n"
        finally:
            if not worker.done():
                worker.cancel()

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    document = await db.scalar(select(Document).where(Document.id == document_id, Document.owner_id == user.id))
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    stored_path = Path(document.storage_path)
    await db.delete(document)
    await db.commit()
    stored_path.unlink(missing_ok=True)
