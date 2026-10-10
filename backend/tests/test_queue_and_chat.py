import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from app.db import SessionLocal, engine
from app.models import ChatMessage, Conversation, Document, TranslationJob, User
from app.retrieval import chunk_text, retrieve_relevant_chunks
from app.security import hash_password
from app.worker import process_next_translation_job


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def test_users():
    async with SessionLocal() as db:
        user_a = User(
            id=uuid4(),
            username=f"user_a_{uuid4().hex[:6]}",
            email=f"user_a_{uuid4().hex[:6]}@example.com",
            name="User A",
            password_hash=hash_password("Pass123!"),
        )
        user_b = User(
            id=uuid4(),
            username=f"user_b_{uuid4().hex[:6]}",
            email=f"user_b_{uuid4().hex[:6]}@example.com",
            name="User B",
            password_hash=hash_password("Pass123!"),
        )
        db.add_all([user_a, user_b])
        await db.commit()
        await db.refresh(user_a)
        await db.refresh(user_b)
        return user_a, user_b


@pytest.mark.asyncio
async def test_upload_uniqueness_and_independent_records(test_users):
    user_a, _ = test_users
    async with SessionLocal() as db:
        # Simulate two uploads of identical text and filename
        filename = "patient_cbc_report.pdf"
        identical_text = "White Blood Cell count: 12.5 x 10^3/uL (Elevated). Platelets: 250 x 10^3/uL."
        
        doc1 = Document(
            id=uuid4(),
            owner_id=user_a.id,
            original_filename=filename,
            media_type="application/pdf",
            storage_path=f"uploads/{user_a.id}_{uuid4().hex}.pdf",
            extracted_text=identical_text,
            summary="Patient exhibits elevated white blood cell count indicating possible infection.",
            summary_method="llm_llama3.2:3b",
            processing_status="translating",
            translations_status={"hi": "pending", "mr": "pending", "ta": "pending"},
        )
        doc2 = Document(
            id=uuid4(),
            owner_id=user_a.id,
            original_filename=filename,
            media_type="application/pdf",
            storage_path=f"uploads/{user_a.id}_{uuid4().hex}.pdf",
            extracted_text=identical_text,
            summary="Patient exhibits elevated white blood cell count indicating possible infection.",
            summary_method="llm_llama3.2:3b",
            processing_status="translating",
            translations_status={"hi": "pending", "mr": "pending", "ta": "pending"},
        )
        db.add_all([doc1, doc2])

        # Enqueue translation jobs for both documents
        jobs_doc1 = [
            TranslationJob(id=uuid4(), document_id=doc1.id, language="hi", order_index=0, status="pending"),
            TranslationJob(id=uuid4(), document_id=doc1.id, language="mr", order_index=1, status="pending"),
            TranslationJob(id=uuid4(), document_id=doc1.id, language="ta", order_index=2, status="pending"),
        ]
        jobs_doc2 = [
            TranslationJob(id=uuid4(), document_id=doc2.id, language="hi", order_index=0, status="pending"),
            TranslationJob(id=uuid4(), document_id=doc2.id, language="mr", order_index=1, status="pending"),
            TranslationJob(id=uuid4(), document_id=doc2.id, language="ta", order_index=2, status="pending"),
        ]
        db.add_all(jobs_doc1 + jobs_doc2)
        await db.commit()

        # Assert uniqueness
        assert doc1.id != doc2.id
        assert doc1.storage_path != doc2.storage_path
        
        # Verify both documents exist in database independently
        fetched1 = await db.get(Document, doc1.id)
        fetched2 = await db.get(Document, doc2.id)
        assert fetched1 is not None and fetched2 is not None
        assert fetched1.id != fetched2.id


@pytest.mark.asyncio
async def test_translation_queue_strict_ordering_and_serialization(monkeypatch, test_users):
    user_a, _ = test_users
    execution_order = []

    def mock_translate_summary(text: str, language: str) -> str:
        execution_order.append(language)
        return f"Translated [{language}]: {text[:20]}"

    monkeypatch.setattr("app.worker.translate_summary", mock_translate_summary)

    async with SessionLocal() as db:
        await db.execute(delete(TranslationJob))
        await db.commit()

        # Create Document A
        doc_a = Document(
            id=uuid4(),
            owner_id=user_a.id,
            original_filename="doc_a.pdf",
            media_type="application/pdf",
            storage_path=f"uploads/{user_a.id}_{uuid4().hex}.pdf",
            extracted_text="Doc A content",
            summary="Doc A summary",
            processing_status="translating",
            translations_status={"hi": "pending", "mr": "pending", "ta": "pending"},
        )
        db.add(doc_a)
        await db.commit()

        jobs_a = [
            TranslationJob(id=uuid4(), document_id=doc_a.id, language="hi", order_index=0, status="pending"),
            TranslationJob(id=uuid4(), document_id=doc_a.id, language="mr", order_index=1, status="pending"),
            TranslationJob(id=uuid4(), document_id=doc_a.id, language="ta", order_index=2, status="pending"),
        ]
        db.add_all(jobs_a)
        await db.commit()

        # Let a tiny time elapse so created_at distinguishes Doc A from Doc B
        await asyncio.sleep(0.01)

        # Create Document B
        doc_b = Document(
            id=uuid4(),
            owner_id=user_a.id,
            original_filename="doc_b.pdf",
            media_type="application/pdf",
            storage_path=f"uploads/{user_a.id}_{uuid4().hex}.pdf",
            extracted_text="Doc B content",
            summary="Doc B summary",
            processing_status="translating",
            translations_status={"hi": "pending", "mr": "pending", "ta": "pending"},
        )
        db.add(doc_b)
        await db.commit()

        jobs_b = [
            TranslationJob(id=uuid4(), document_id=doc_b.id, language="hi", order_index=0, status="pending"),
            TranslationJob(id=uuid4(), document_id=doc_b.id, language="mr", order_index=1, status="pending"),
            TranslationJob(id=uuid4(), document_id=doc_b.id, language="ta", order_index=2, status="pending"),
        ]
        db.add_all(jobs_b)
        await db.commit()

    # Process all jobs sequentially using process_next_translation_job()
    processed_count = 0
    for _ in range(10):
        had_work = await process_next_translation_job()
        if not had_work:
            break
        processed_count += 1

    # Verify all 6 jobs processed
    assert processed_count == 6

    # Verify strict serialization order: Record A (hi -> mr -> ta) then Record B (hi -> mr -> ta)
    assert execution_order == ["hi", "mr", "ta", "hi", "mr", "ta"]

    # Verify Document A final state
    async with SessionLocal() as db:
        doc_a_final = await db.get(Document, doc_a.id)
        assert doc_a_final.hindi_summary is not None
        assert doc_a_final.marathi_summary is not None
        assert doc_a_final.tamil_summary is not None
        assert doc_a_final.translations_status == {
            "hi": "completed",
            "mr": "completed",
            "ta": "completed",
        }
        assert doc_a_final.processing_status == "completed"

        # Verify Document B final state
        doc_b_final = await db.get(Document, doc_b.id)
        assert doc_b_final.hindi_summary is not None
        assert doc_b_final.marathi_summary is not None
        assert doc_b_final.tamil_summary is not None
        assert doc_b_final.translations_status == {
            "hi": "completed",
            "mr": "completed",
            "ta": "completed",
        }
        assert doc_b_final.processing_status == "completed"


@pytest.mark.asyncio
async def test_retranslation_job_enqueuing_and_execution(monkeypatch, test_users):
    user_a, _ = test_users
    monkeypatch.setattr(
        "app.worker.translate_summary",
        lambda text, lang: f"Updated [{lang}] translation",
    )

    async with SessionLocal() as db:
        await db.execute(delete(TranslationJob))
        await db.commit()

        doc = Document(
            id=uuid4(),
            owner_id=user_a.id,
            original_filename="doc_retrans.pdf",
            media_type="application/pdf",
            storage_path=f"uploads/{user_a.id}_{uuid4().hex}.pdf",
            extracted_text="Content",
            summary="Original summary",
            hindi_summary="Old Hindi",
            marathi_summary="Old Marathi",
            tamil_summary="Old Tamil",
            processing_status="completed",
            translations_status={"hi": "completed", "mr": "completed", "ta": "completed"},
        )
        db.add(doc)
        await db.commit()

        # Simulate user requesting manual retranslation for "mr" (Marathi)
        t_status = dict(doc.translations_status)
        t_status["mr"] = "pending"
        doc.translations_status = t_status
        doc.processing_status = "translating"
        retrans_job = TranslationJob(
            id=uuid4(),
            document_id=doc.id,
            language="mr",
            order_index=0,
            status="pending",
            is_manual=True,
        )
        db.add(retrans_job)
        await db.commit()

    # Process retranslation job
    had_work = await process_next_translation_job()
    assert had_work is True

    # Verify Marathi summary was updated and other languages preserved
    async with SessionLocal() as db:
        updated_doc = await db.get(Document, doc.id)
        assert updated_doc.marathi_summary == "Updated [mr] translation"
        assert updated_doc.hindi_summary == "Old Hindi"
        assert updated_doc.tamil_summary == "Old Tamil"
        assert updated_doc.translations_status["mr"] == "completed"
        assert updated_doc.processing_status == "completed"


@pytest.mark.asyncio
async def test_chat_persistence_and_isolation(test_users):
    user_a, user_b = test_users
    async with SessionLocal() as db:
        doc_a = Document(
            id=uuid4(),
            owner_id=user_a.id,
            original_filename="doc_user_a.pdf",
            media_type="application/pdf",
            storage_path=f"uploads/{user_a.id}_{uuid4().hex}.pdf",
            extracted_text="Patient has normal ECG.",
            summary="ECG normal.",
            processing_status="completed",
        )
        db.add(doc_a)
        await db.commit()

        # Create conversation for User A
        conv_a = Conversation(id=uuid4(), document_id=doc_a.id, user_id=user_a.id)
        db.add(conv_a)
        await db.commit()

        # Add messages
        msg1 = ChatMessage(id=uuid4(), conversation_id=conv_a.id, role="user", content="Is ECG normal?")
        msg2 = ChatMessage(id=uuid4(), conversation_id=conv_a.id, role="assistant", content="Yes, the ECG is normal.")
        db.add_all([msg1, msg2])
        await db.commit()

        # Verify messages persisted and in order
        conv_fetched = await db.scalar(
            select(Conversation)
            .where(Conversation.id == conv_a.id)
            .options(selectinload(Conversation.messages))
        )
        assert len(conv_fetched.messages) == 2
        assert conv_fetched.messages[0].role == "user"
        assert conv_fetched.messages[0].content == "Is ECG normal?"
        assert conv_fetched.messages[1].role == "assistant"
        assert conv_fetched.messages[1].content == "Yes, the ECG is normal."

        # Verify User B does not own or have access to Doc A's conversation
        unauthorized_conv = await db.scalar(
            select(Conversation).where(
                Conversation.document_id == doc_a.id,
                Conversation.user_id == user_b.id,
            )
        )
        assert unauthorized_conv is None


def test_bm25_retrieval():
    long_document = """
    Patient Information: John Doe, 45-year-old male.
    Chief Complaint: Persistent headache and fatigue for three weeks.
    
    Cardiovascular Examination:
    Blood pressure 128/82 mmHg. Heart sounds S1 and S2 present, regular rate and rhythm.
    No murmurs, rubs, or gallops identified on auscultation.
    
    Neurological Assessment:
    Cranial nerves II through XII grossly intact. Normal gait and stance.
    Deep tendon reflexes 2+ bilaterally in upper and lower extremities.
    
    Laboratory Analysis:
    Complete Blood Count: Hemoglobin 14.2 g/dL, Hematocrit 42%.
    White blood cell count normal at 6,500 /uL. Platelets normal at 220,000 /uL.
    Serum creatinine 0.9 mg/dL, Blood Urea Nitrogen 15 mg/dL.
    Fast blood glucose 92 mg/dL.
    
    Impression and Plan:
    Tension headache secondary to occupational stress and sleep disruption.
    Advise lifestyle modifications, sleep hygiene, and PRN acetaminophen.
    """
    
    # Query specific to blood count
    chunks = retrieve_relevant_chunks(long_document, "What is the white blood cell count?", top_k=2, max_total_words=50)
    assert len(chunks) > 0
    # Relevant chunk should contain laboratory analysis
    found_lab = any("White blood cell count" in c for c in chunks)
    assert found_lab is True
    
    # Query specific to cardiac
    cardiac_chunks = retrieve_relevant_chunks(long_document, "Are there any heart murmurs or abnormal sounds?", top_k=2, max_total_words=50)
    assert len(cardiac_chunks) > 0
    found_cardiac = any("Cardiovascular Examination" in c or "Heart sounds" in c for c in cardiac_chunks)
    assert found_cardiac is True
