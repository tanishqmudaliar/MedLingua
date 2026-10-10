from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_.-]+$")
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserLogin(BaseModel):
    username: str
    password: str


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    name: str
    email: EmailStr


class DocumentTranslationCache(BaseModel):
    summary: str | None = None
    extracted_text: str | None = None
    extracted_chunks: list[str] = Field(default_factory=list)


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    original_filename: str
    media_type: str
    extracted_text: str
    summary: str | None
    summary_method: str | None = None
    processing_status: str = "uploaded"
    error_message: str | None = None
    hindi_summary: str | None = None
    marathi_summary: str | None = None
    tamil_summary: str | None = None
    translations_status: dict[str, str] = Field(default_factory=dict)
    translations: dict[str, DocumentTranslationCache] = Field(default_factory=dict)
    created_at: datetime


class DocumentTranslationRequest(BaseModel):
    language: Literal["hi", "mr", "ta"]
    force: bool = False


class DocumentTranslationRead(BaseModel):
    language: Literal["hi", "mr", "ta"]
    summary: str | None
    extracted_text: str


class RetranslateRequest(BaseModel):
    language: Literal["hi", "mr", "ta"]


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: str
    content: str
    created_at: datetime


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    messages: list[ChatMessageRead] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
