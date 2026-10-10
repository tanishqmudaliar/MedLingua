import json
import logging
import re
import urllib.request
from threading import RLock
from typing import Callable, Literal

from .config import get_settings

TargetLanguage = Literal["hi", "mr", "ta"]

LANGUAGES: dict[TargetLanguage, str] = {
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
}

# AI4Bharat IndicTrans2 Language Tag Mapping
INDICTRANS2_TAGS: dict[TargetLanguage, str] = {
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "ta": "tam_Taml",
}
SRC_LANG = "eng_Latn"
M2M100_MODEL_NAME = "facebook/m2m100_418M"

_MODEL_LOCK = RLock()
_MODEL = None
_TOKENIZER = None
_PROCESSOR = None
_MODEL_DEVICE = "cpu"
_ENGINE_TYPE = "indictrans2"  # "indictrans2" or "m2m100"
logger = logging.getLogger(__name__)


class TranslationModelUnavailable(RuntimeError):
    pass


def _select_device(cuda_available: bool) -> str:
    return "cuda" if cuda_available else "cpu"


def _unload_ollama_to_free_vram() -> None:
    """Releases Ollama LLM from GPU memory so translation model has maximum available VRAM."""
    settings = get_settings()
    try:
        req = urllib.request.Request(
            f"{settings.llm_api_base}/api/generate",
            data=json.dumps({"model": settings.llm_model, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=2):
            pass
        logger.info("Unloaded Ollama LLM to free VRAM for translation model.")
    except Exception:
        pass


def _load_indictrans2():
    global _MODEL, _TOKENIZER, _PROCESSOR, _MODEL_DEVICE, _ENGINE_TYPE
    if _MODEL is not None and _TOKENIZER is not None and _PROCESSOR is not None:
        return _MODEL, _TOKENIZER, _PROCESSOR

    import torch
    import transformers
    import transformers.tokenization_utils
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Compatibility shim for IndicTransToolkit with modern Transformers
    if not hasattr(transformers.tokenization_utils, "PreTrainedTokenizerBase"):
        transformers.tokenization_utils.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase

    from IndicTransToolkit.processor import IndicProcessor

    settings = get_settings()
    model_name = settings.indictrans_model_name

    # Hardware device resolution
    cuda_available = torch.cuda.is_available()
    if settings.indictrans_device == "cpu":
        device = "cpu"
    elif settings.indictrans_device == "cuda":
        device = "cuda" if cuda_available else "cpu"
    else:  # "auto"
        device = "cuda" if cuda_available else "cpu"

    if device == "cuda":
        _unload_ollama_to_free_vram()

    token = settings.hf_token
    logger.info("Loading IndicTrans2 (%s) on %s...", model_name, device.upper())

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=token,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=token,
        low_cpu_mem_usage=True,
    )

    if device == "cuda":
        model.half()
        model.to("cuda")
    else:
        model.to("cpu")

    model.eval()
    processor = IndicProcessor(inference=True)

    _MODEL = model
    _TOKENIZER = tokenizer
    _PROCESSOR = processor
    _MODEL_DEVICE = device
    _ENGINE_TYPE = "indictrans2"
    logger.info("IndicTrans2 successfully loaded on %s", device.upper())
    return _MODEL, _TOKENIZER, _PROCESSOR


def _load_m2m100():
    global _MODEL, _TOKENIZER, _PROCESSOR, _MODEL_DEVICE, _ENGINE_TYPE
    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    settings = get_settings()
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available and settings.indictrans_device != "cpu" else "cpu"
    if device == "cuda":
        _unload_ollama_to_free_vram()

    logger.info("Loading fallback translation model (%s) on %s...", M2M100_MODEL_NAME, device.upper())
    tokenizer = M2M100Tokenizer.from_pretrained(M2M100_MODEL_NAME)
    model = M2M100ForConditionalGeneration.from_pretrained(M2M100_MODEL_NAME)

    if device == "cuda":
        model.half()
        model.to("cuda")
    else:
        model.to("cpu")

    model.eval()
    _MODEL = model
    _TOKENIZER = tokenizer
    _PROCESSOR = None
    _MODEL_DEVICE = device
    _ENGINE_TYPE = "m2m100"
    logger.info("M2M100 successfully loaded as fallback translation engine on %s", device.upper())
    return _MODEL, _TOKENIZER, None


def _load_translation_engine():
    global _MODEL, _TOKENIZER, _PROCESSOR, _MODEL_DEVICE, _ENGINE_TYPE
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER, _PROCESSOR, _ENGINE_TYPE

    with _MODEL_LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return _MODEL, _TOKENIZER, _PROCESSOR, _ENGINE_TYPE

        settings = get_settings()
        # If HF_TOKEN is present, try IndicTrans2
        if settings.hf_token:
            try:
                m, t, p = _load_indictrans2()
                _ENGINE_TYPE = "indictrans2"
                return m, t, p, _ENGINE_TYPE
            except Exception as exc:
                logger.warning("IndicTrans2 loading failed with provided HF_TOKEN (%s). Falling back to M2M100.", exc)
        else:
            # Try IndicTrans2 first in case cached or ungated
            try:
                m, t, p = _load_indictrans2()
                _ENGINE_TYPE = "indictrans2"
                return m, t, p, _ENGINE_TYPE
            except Exception as exc:
                logger.info(
                    "Notice: IndicTrans2 is a gated Hugging Face repo (requires token). "
                    "Auto-fallback to public model '%s'. "
                    "(To use IndicTrans2: request access at https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M "
                    "and set HF_TOKEN in backend/.env)",
                    M2M100_MODEL_NAME,
                )

        try:
            m, t, p = _load_m2m100()
            return m, t, p, _ENGINE_TYPE
        except Exception as m2m_exc:
            logger.exception("Fallback translation model loading failed: %s", m2m_exc)
            raise TranslationModelUnavailable(
                f"Translation engine unavailable: {m2m_exc}. Please check your internet connection."
            ) from m2m_exc


def translate_summary(text: str, language: TargetLanguage) -> str:
    """Translates clinical summary into Hindi, Marathi, or Tamil using IndicTrans2 or M2M100 fallback."""
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported translation language: {language}")
    if not text.strip():
        return ""

    with _MODEL_LOCK:
        model, tokenizer, processor, engine_type = _load_translation_engine()
        import torch

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        translated_paragraphs = []

        if engine_type == "indictrans2" and processor is not None:
            tgt_lang = INDICTRANS2_TAGS[language]
            for paragraph in paragraphs:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
                if not sentences:
                    continue

                batch = processor.preprocess_batch(sentences, src_lang=SRC_LANG, tgt_lang=tgt_lang)
                inputs = tokenizer(
                    batch,
                    padding="longest",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(_MODEL_DEVICE)

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        num_beams=1,
                        max_length=256,
                        repetition_penalty=1.1,
                    )

                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                postprocessed = processor.postprocess_batch(decoded, lang=tgt_lang)
                translated_paragraphs.append(" ".join(postprocessed))
        else:
            # M2M100 Fallback Translation
            tokenizer.src_lang = "en"
            for paragraph in paragraphs:
                inputs = tokenizer(
                    paragraph,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(_MODEL_DEVICE)

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.get_lang_id(language),
                        max_new_tokens=256,
                        repetition_penalty=1.1,
                    )

                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                translated_paragraphs.append(decoded)

        return "\n\n".join(translated_paragraphs)


def _split_into_chunks(text: str) -> list[str]:
    """Splits document text into readable sections/paragraphs suitable for chunk-by-chunk translation."""
    chunks: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        cleaned = text.strip()
        return [cleaned] if cleaned else []

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) <= 120:
            chunks.append(paragraph)
        else:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
            cur_chunk: list[str] = []
            cur_len = 0
            for sentence in sentences:
                s_len = len(sentence.split())
                if cur_chunk and (cur_len + s_len > 80):
                    chunks.append(" ".join(cur_chunk))
                    cur_chunk = [sentence]
                    cur_len = s_len
                else:
                    cur_chunk.append(sentence)
                    cur_len += s_len
            if cur_chunk:
                chunks.append(" ".join(cur_chunk))
    return chunks or [text.strip()]


def translate_text(
    text: str,
    language: TargetLanguage,
    on_progress: Callable[[int, int, str], None] | None = None,
    cached_chunks: list[str] | None = None,
) -> str:
    """Translates full extracted document text chunk-by-chunk with real-time section progress callbacks."""
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported translation language: {language}")
    if not text.strip():
        return ""

    chunks = _split_into_chunks(text)
    total_chunks = len(chunks)
    translated: list[str] = list(cached_chunks or [])

    if len(translated) > total_chunks:
        translated = []

    if on_progress:
        on_progress(len(translated), total_chunks, "")

    start_idx = len(translated)
    for idx, chunk in enumerate(chunks[start_idx:], start=start_idx + 1):
        translated_chunk = translate_summary(chunk, language)
        translated.append(translated_chunk)
        if on_progress:
            on_progress(idx, total_chunks, translated_chunk)

    return "\n\n".join(translated)
