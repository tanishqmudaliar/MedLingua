import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import SessionLocal
from .models import Document, TranslationJob
from .translation import TranslationModelUnavailable, translate_summary

logger = logging.getLogger(__name__)

_WORKER_RUNNING = False
_WORKER_LOCK = asyncio.Lock()


async def process_next_translation_job() -> bool:
    """
    Picks and processes exactly ONE translation job in strict deterministic order:
    Record A (hi -> mr -> ta) -> Record B (hi -> mr -> ta).
    Returns True if a job was processed, False if queue is idle.
    """
    async with SessionLocal() as db:
        # Check if there is an active document being processed
        active_job = await db.scalar(
            select(TranslationJob)
            .where(TranslationJob.status == "processing")
            .order_by(TranslationJob.created_at.asc(), TranslationJob.order_index.asc())
        )
        if active_job:
            job = active_job
        else:
            # Prioritize interactive manual retranslations requested by users
            manual_pending = await db.scalar(
                select(TranslationJob)
                .where(TranslationJob.status == "pending", TranslationJob.is_manual == True)
                .order_by(TranslationJob.created_at.asc())
            )
            if manual_pending:
                job = manual_pending
            else:
                # 1. Find the oldest document that has pending batch jobs
                # To ensure Record A finishes hi -> mr -> ta before Record B starts:
                earliest_pending = await db.scalar(
                    select(TranslationJob)
                    .where(TranslationJob.status == "pending")
                    .order_by(TranslationJob.created_at.asc(), TranslationJob.order_index.asc())
                )
                if not earliest_pending:
                    return False

                # Lock this document's next pending job in order_index sequence
                job = await db.scalar(
                    select(TranslationJob)
                    .where(
                        TranslationJob.document_id == earliest_pending.document_id,
                        TranslationJob.status == "pending",
                    )
                    .order_by(TranslationJob.order_index.asc())
                )
                if not job:
                    job = earliest_pending

        # Mark job as processing
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        document = await db.get(Document, job.document_id)
        if document:
            t_status = dict(document.translations_status or {})
            t_status[job.language] = "translating"
            document.translations_status = t_status
            document.processing_status = "translating"
        await db.commit()

        # Execute translation outside of DB lock
        lang = job.language
        summary_to_translate = document.summary if document else ""
        error_msg = None
        translated_text = None

        if not summary_to_translate:
            error_msg = "No English summary available to translate"
        else:
            try:
                translated_text = await asyncio.to_thread(
                    translate_summary, summary_to_translate, lang
                )
            except TranslationModelUnavailable as exc:
                error_msg = str(exc)
                logger.warning("Translation model unavailable for %s: %s", lang, exc)
            except Exception as exc:
                error_msg = f"Translation failed: {exc}"
                logger.exception("Unexpected error translating job %s (%s)", job.id, lang)

        # Update record and job
        async with SessionLocal() as update_db:
            job_ref = await update_db.get(TranslationJob, job.id)
            doc_ref = await update_db.get(Document, job.document_id)
            if job_ref and doc_ref:
                t_status = dict(doc_ref.translations_status or {})
                if translated_text:
                    job_ref.status = "completed"
                    job_ref.completed_at = datetime.now(timezone.utc)
                    job_ref.error_message = None
                    t_status[lang] = "completed"

                    if lang == "hi":
                        doc_ref.hindi_summary = translated_text
                    elif lang == "mr":
                        doc_ref.marathi_summary = translated_text
                    elif lang == "ta":
                        doc_ref.tamil_summary = translated_text
                else:
                    job_ref.status = "failed"
                    job_ref.completed_at = datetime.now(timezone.utc)
                    job_ref.error_message = error_msg
                    t_status[lang] = "failed"

                doc_ref.translations_status = t_status

                # Check if all initial jobs for this document have reached terminal state
                remaining = await update_db.scalars(
                    select(TranslationJob).where(
                        TranslationJob.document_id == doc_ref.id,
                        TranslationJob.status.in_(["pending", "processing"]),
                    )
                )
                if not list(remaining):
                    doc_ref.processing_status = "completed"

                await update_db.commit()

        return True


async def translation_worker_loop():
    """Continuous async worker loop processing serialized jobs."""
    global _WORKER_RUNNING
    if _WORKER_RUNNING:
        return
    _WORKER_RUNNING = True
    logger.info("Translation worker loop started.")

    # Recover any crashed 'processing' jobs on worker startup
    try:
        async with SessionLocal() as db:
            crashed = await db.scalars(
                select(TranslationJob).where(TranslationJob.status == "processing")
            )
            for c_job in crashed:
                c_job.status = "pending"
            await db.commit()
    except Exception as exc:
        logger.warning("Failed to recover interrupted jobs: %s", exc)

    while True:
        try:
            had_work = await process_next_translation_job()
            if not had_work:
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            logger.info("Translation worker loop canceled.")
            break
        except Exception as exc:
            logger.error("Error in translation worker loop: %s", exc)
            await asyncio.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(translation_worker_loop())
