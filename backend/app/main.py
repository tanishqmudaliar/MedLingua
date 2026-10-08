from pathlib import Path
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from sqlalchemy import delete, desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .db import get_db
from .dependencies import get_current_user
from .models import Document, User
from .ocr import extract_image_text, extract_pdf_text
from .schemas import DocumentRead, UserCreate, UserLogin, UserRead
from .security import create_access_token, hash_password, verify_password
from .summary import summarize_text

settings = get_settings()
app = FastAPI(title="MedLingua API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/signup", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def signup(payload: UserCreate, response: Response, db: AsyncSession = Depends(get_db)) -> User:
    existing = await db.scalar(select(User).where(or_(User.username == payload.username, User.email == payload.email.lower())))
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
    response.set_cookie("access_token", create_access_token(user.id), httponly=True, samesite="lax", max_age=settings.jwt_expire_minutes * 60)
    return user


@app.post("/auth/login", response_model=UserRead)
async def login(payload: UserLogin, response: Response, db: AsyncSession = Depends(get_db)) -> User:
    user = await db.scalar(select(User).where(User.username == payload.username))
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found. Sign up to create your account.")
    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect password")
    response.set_cookie("access_token", create_access_token(user.id), httponly=True, samesite="lax", max_age=settings.jwt_expire_minutes * 60)
    return user


@app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> None:
    response.delete_cookie("access_token")


@app.get("/auth/me", response_model=UserRead)
async def me(user: User = Depends(get_current_user)) -> User:
    return user


@app.post("/documents", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Document:
    if file.content_type != "application/pdf" and file.content_type not in IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Only PDF, PNG, JPG, JPEG, WEBP, BMP, or TIFF files are supported")
    data = await file.read(settings.max_upload_size_bytes + 1)
    if len(data) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds the {settings.max_upload_size_mb} MB limit")
    suffix = Path(file.filename or "upload").suffix.lower() or ".bin"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = settings.upload_dir / f"{user.id}_{uuid4().hex}{suffix}"
    stored_path.write_bytes(data)
    try:
        text = extract_pdf_text(stored_path, settings.max_pdf_pages) if file.content_type == "application/pdf" else extract_image_text(data)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        stored_path.unlink(missing_ok=True)
        if isinstance(exc, (ImportError, OSError)):
            detail = "PaddleOCR could not load. Check the pinned PaddleOCR/PaddlePaddle dependencies and the Python environment."
            raise HTTPException(status_code=503, detail=detail) from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    summary = await run_in_threadpool(summarize_text, text)
    document = Document(
        owner_id=user.id,
        original_filename=file.filename or "upload",
        media_type=file.content_type,
        storage_path=str(stored_path),
        extracted_text=text,
        summary=summary or None,
    )
    db.add(document)
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
