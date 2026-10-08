import re
import logging
from threading import RLock
from typing import Callable, Literal

TargetLanguage = Literal["hi", "mr", "ta"]

LANGUAGES: dict[TargetLanguage, str] = {
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
}
MODEL_NAME = "facebook/m2m100_418M"
_CHUNK_TOKEN_LIMIT = 256
_MODEL_LOCK = RLock()
_MODEL = None
_TOKENIZER = None
_MODEL_DEVICE = "cpu"
logger = logging.getLogger(__name__)


class TranslationModelUnavailable(RuntimeError):
    pass


def _select_device(cuda_available: bool) -> str:
    return "cuda" if cuda_available else "cpu"


def _load_model():
    global _MODEL, _TOKENIZER, _MODEL_DEVICE
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    with _MODEL_LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return _MODEL, _TOKENIZER
        try:
            import torch
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

            device = _select_device(torch.cuda.is_available())
            tokenizer = M2M100Tokenizer.from_pretrained(
                MODEL_NAME, local_files_only=True
            )
            model = M2M100ForConditionalGeneration.from_pretrained(
                MODEL_NAME, local_files_only=True
            )
            model.to(device)
            model.eval()
            _MODEL = model
            _TOKENIZER = tokenizer
            _MODEL_DEVICE = device
            logger.info("Loaded local translation model on %s", device.upper())
        except (ImportError, OSError) as exc:
            _MODEL = None
            _TOKENIZER = None
            raise TranslationModelUnavailable(
                "Local translation is unavailable. Install backend requirements "
                "and, from the backend folder, run "
                "`python -m app.download_translation_model` once, then retry. "
                "Report text is not sent to an external service."
            ) from exc
    return _MODEL, _TOKENIZER


def _split_into_chunks(text: str, tokenizer) -> list[str]:
    chunks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current_ids: list[int] = []
        for sentence in sentences:
            sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
            if not sentence_ids:
                continue
            for offset in range(0, len(sentence_ids), _CHUNK_TOKEN_LIMIT):
                piece = sentence_ids[offset:offset + _CHUNK_TOKEN_LIMIT]
                if current_ids and len(current_ids) + len(piece) > _CHUNK_TOKEN_LIMIT:
                    chunks.append(tokenizer.decode(current_ids, skip_special_tokens=True))
                    current_ids = []
                current_ids.extend(piece)
        if current_ids:
            chunks.append(tokenizer.decode(current_ids, skip_special_tokens=True))
    return chunks


def translate_text(
    text: str,
    language: TargetLanguage,
    on_progress: Callable[[int, int, str], None] | None = None,
    cached_chunks: list[str] | None = None,
) -> str:
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported translation language: {language}")
    if not text.strip():
        return ""

    import torch

    with _MODEL_LOCK:
        model, tokenizer = _load_model()
        tokenizer.src_lang = "en"
        chunks = _split_into_chunks(text, tokenizer)
        translated = list(cached_chunks or [])
        if len(translated) > len(chunks):
            raise ValueError("Cached translation does not match the source text")
        if on_progress:
            on_progress(len(translated), len(chunks), "")
        for index, chunk in enumerate(chunks[len(translated):], start=len(translated) + 1):
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=_CHUNK_TOKEN_LIMIT + 2,
            )
            encoded = {
                key: value.to(_MODEL_DEVICE) for key, value in encoded.items()
            }
            input_length = int(encoded["input_ids"].shape[-1])
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.get_lang_id(language),
                    max_new_tokens=min(512, max(64, int(input_length * 1.5) + 32)),
                    num_beams=1,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1,
                )
            translated_chunk = tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()
            translated.append(translated_chunk)
            if on_progress:
                on_progress(index, len(chunks), translated_chunk)
    return "\n\n".join(part for part in translated if part)
