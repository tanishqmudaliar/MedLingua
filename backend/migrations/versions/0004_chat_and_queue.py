"""hardware, chat, and translation queue schema

Revision ID: 0004_hardware_chat_translation_queue
Revises: 0003_document_translations
Create Date: 2026-10-10
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_chat_and_queue"
down_revision = "0003_document_translations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns to documents
    op.add_column("documents", sa.Column("processing_status", sa.String(32), server_default="uploaded", nullable=False))
    op.add_column("documents", sa.Column("summary_method", sa.String(32), nullable=True))
    op.add_column("documents", sa.Column("error_message", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("hindi_summary", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("marathi_summary", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("tamil_summary", sa.Text(), nullable=True))
    op.add_column(
        "documents",
        sa.Column(
            "translations_status",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{\"hi\": \"pending\", \"mr\": \"pending\", \"ta\": \"pending\"}'::jsonb"),
            nullable=False,
        ),
    )

    # Create translation_jobs table
    op.create_table(
        "translation_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("language", sa.String(10), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("is_manual", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_translation_jobs_document_id", "translation_jobs", ["document_id"])
    op.create_index("ix_translation_jobs_status", "translation_jobs", ["status"])

    # Create conversations table
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_conversations_document_id", "conversations", ["document_id"], unique=True)
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])

    # Create chat_messages table
    op.create_table(
        "chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_chat_messages_conversation_id", "chat_messages", ["conversation_id"])
    op.create_index("ix_chat_messages_created_at", "chat_messages", ["created_at"])


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("conversations")
    op.drop_table("translation_jobs")
    op.drop_column("documents", "translations_status")
    op.drop_column("documents", "tamil_summary")
    op.drop_column("documents", "marathi_summary")
    op.drop_column("documents", "hindi_summary")
    op.drop_column("documents", "error_message")
    op.drop_column("documents", "summary_method")
    op.drop_column("documents", "processing_status")
