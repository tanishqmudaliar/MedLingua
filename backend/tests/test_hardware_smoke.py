import asyncio
import json
import logging
import time
import urllib.request
import pytest
import pytest_asyncio

from app.config import get_settings
from app.llm import check_gpu_diagnostics, generate_document_summary, stream_chatbot_response


CLINICAL_TEST_RECORD = """CLINICAL REPORT
Patient Name: Ramesh Patel, Age: 58, Gender: Male
Chief Complaint: 3-day history of exertional chest pressure and mild dyspnea on climbing stairs.
Past Medical History: Type 2 Diabetes Mellitus diagnosed 6 years ago (last HbA1c 7.8%), Essential Hypertension.
Physical Examination: BP 142/88 mmHg, HR 82 bpm regular, SpO2 98% on room air.
Cardiovascular: Normal S1 and S2, no cardiac murmurs, rubs, or gallops.
Respiratory: Bilateral vesicular breath sounds, no crackles or wheezing.
Investigations:
- High-sensitivity Troponin I: 0.02 ng/mL (Normal < 0.04 ng/mL, negative for acute myocardial necrosis).
- 12-Lead ECG: Normal sinus rhythm at 80 bpm, non-specific T-wave flattening in lead III, no acute ST-elevation.
- Lipid Panel: Total Cholesterol 228 mg/dL, LDL 146 mg/dL, HDL 41 mg/dL, Triglycerides 205 mg/dL.
Assessment:
1. Exertional chest pressure - low intermediate risk for stable angina, acute coronary syndrome ruled out.
2. Suboptimally controlled Type 2 Diabetes Mellitus and dyslipidemia.
Plan:
1. Schedule outpatient exercise stress echocardiography within 72 hours.
2. Initiate Atorvastatin 20 mg PO daily at bedtime.
3. Titrate Metformin to 1000 mg PO twice daily with meals.
4. Prescribe sublingual Nitroglycerin 0.4 mg PRN for acute chest discomfort.
5. Strict instructions to present to the emergency department if chest pain occurs at rest or exceeds 15 minutes.
"""


def test_hardware_gpu_diagnostics():
    diagnostics = check_gpu_diagnostics()
    print("\n--- HARDWARE & GPU DIAGNOSTICS ---")
    print(f"GPU Detected: {diagnostics.get('gpu_detected')}")
    print(f"VRAM Used: {diagnostics.get('vram_used_mb')} MB / {diagnostics.get('vram_total_mb')} MB")
    
    assert diagnostics.get("gpu_detected") is True, "NVIDIA GPU not detected via nvidia-smi"
    assert diagnostics.get("vram_total_mb", 0) >= 3000, "Expected at least ~4 GB dedicated VRAM"


def test_live_llm_summarization_and_layer_allocation():
    settings = get_settings()
    diag_before = check_gpu_diagnostics()
    vram_before = diag_before.get("vram_used_mb", 0)

    start_time = time.perf_counter()
    summary, method = generate_document_summary(CLINICAL_TEST_RECORD)
    duration = time.perf_counter() - start_time

    diag_after = check_gpu_diagnostics()
    vram_after = diag_after.get("vram_used_mb", 0)

    # Check active models from Ollama
    req = urllib.request.Request(f"{settings.llm_api_base}/api/ps")
    ollama_models = []
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            ollama_models = data.get("models", [])
    except Exception as exc:
        pytest.fail(f"Could not connect to Ollama: {exc}")

    active_model = next((m for m in ollama_models if "llama3.2" in m.get("name", "").lower()), None)
    
    # Calculate layer allocation
    size_total = active_model.get("size", 0) if active_model else 0
    size_vram = active_model.get("size_vram", 0) if active_model else 0
    cpu_offload_pct = 0.0
    if size_total > 0:
        cpu_offload_pct = max(0.0, (1.0 - (size_vram / size_total)) * 100)

    print("\n================ LIVE HARDWARE BENCHMARK: LLM SUMMARIZATION ================")
    print(f"Model: {settings.llm_model} | Context: {settings.llm_num_ctx} | Max Tokens: {settings.llm_max_output_tokens}")
    print(f"Summary Method: {method}")
    print(f"Inference Latency: {duration:.2f} seconds")
    print(f"Summary Word Count: {len(summary.split())} words")
    print(f"VRAM Before: {vram_before} MB -> VRAM After: {vram_after} MB")
    print(f"Model Total Size: {size_total / (1024**3):.2f} GB | Model in VRAM: {size_vram / (1024**3):.2f} GB")
    print(f"CPU Layer Offload: {cpu_offload_pct:.1f}%")
    print("-----------------------------------------------------------------------------")
    print(f"Generated Summary Snippet:\n{summary[:300]}...")
    print("=============================================================================\n")

    assert method == "llm", f"Expected LLM summarization, got {method}"
    assert len(summary.split()) >= 30, "Generated summary is too short"
    assert "chest" in summary.lower() or "angina" in summary.lower(), "Summary missing key clinical symptom"
    assert size_vram > 0, "Model is not loaded into VRAM"
    assert cpu_offload_pct == 0.0, f"Expected 0% CPU layer offload, got {cpu_offload_pct:.1f}%"


@pytest.mark.asyncio
async def test_live_chat_streaming_latency():
    settings = get_settings()
    user_query = "What medication was started for the patient's cholesterol or heart?"
    
    start_time = time.perf_counter()
    first_token_time = None
    streamed_tokens = []

    async for token in stream_chatbot_response(
        extracted_text=CLINICAL_TEST_RECORD,
        english_summary="Patient Ramesh Patel presented with exertional chest pressure. Workup negative for acute MI. Started on Atorvastatin and Nitroglycerin.",
        vernacular_summaries={"hi": None, "mr": None, "ta": None},
        metadata={"filename": "Ramesh_Patel_Report.pdf"},
        history=[],
        user_query=user_query,
    ):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        streamed_tokens.append(token)

    total_duration = time.perf_counter() - start_time
    ttft = (first_token_time - start_time) if first_token_time else 0.0
    full_answer = "".join(streamed_tokens).strip()

    print("\n================ LIVE HARDWARE BENCHMARK: CHAT STREAMING ================")
    print(f"User Query: {user_query}")
    print(f"Time to First Token (TTFT): {ttft:.2f} seconds")
    print(f"Total Streaming Duration: {total_duration:.2f} seconds")
    print(f"Streamed Output Tokens: {len(streamed_tokens)} tokens")
    print("--------------------------------------------------------------------------")
    print(f"Chatbot Response:\n{full_answer}")
    print("==========================================================================\n")

    assert len(streamed_tokens) > 5, "Expected multi-token streamed response"
    assert "atorvastatin" in full_answer.lower(), "Chatbot response did not identify Atorvastatin"
    assert total_duration < 15.0, f"Streaming latency was too slow: {total_duration:.2f}s"
