import torch

from app import translation
from app.translation import LANGUAGES, _split_into_chunks


class FakeTokenizer:
    src_lang = ""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"token{token_id}" for token_id in token_ids)

    def __call__(self, text: str, **kwargs):
        return {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }

    def get_lang_id(self, language: str) -> int:
        return {"hi": 1, "mr": 2, "ta": 3}[language]

    def batch_decode(self, generated, skip_special_tokens: bool = True) -> list[str]:
        return ["translated report"]


class FakeModel:
    def __init__(self) -> None:
        self.target_language = None
        self.generation_options = {}

    def generate(self, **kwargs):
        self.target_language = kwargs["forced_bos_token_id"]
        self.generation_options = {
            "no_repeat_ngram_size": kwargs["no_repeat_ngram_size"],
            "repetition_penalty": kwargs["repetition_penalty"],
        }
        return torch.tensor([[1]])


def test_translation_languages_cover_requested_targets() -> None:
    assert LANGUAGES == {"hi": "Hindi", "mr": "Marathi", "ta": "Tamil"}


def test_translation_chunks_long_text_and_keeps_paragraph_boundaries() -> None:
    tokenizer = FakeTokenizer()
    text = " ".join(f"word{i}" for i in range(850)) + "\n\nA second paragraph."

    chunks = _split_into_chunks(text, tokenizer)

    assert len(chunks) == 5
    assert all(len(tokenizer.encode(chunk)) <= 256 for chunk in chunks)
    assert chunks[-1].endswith("token2")


def test_translation_sets_target_language_and_returns_local_model_output(monkeypatch) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    progress: list[tuple[int, int, str]] = []
    monkeypatch.setattr(translation, "_load_model", lambda: (model, tokenizer))

    result = translation.translate_text(
        "A medical report.", "ta", on_progress=lambda *event: progress.append(event)
    )

    assert tokenizer.src_lang == "en"
    assert model.target_language == 3
    assert model.generation_options == {
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.1,
    }
    assert result == "translated report"
    assert progress == [(0, 1, ""), (1, 1, "translated report")]


def test_translation_resumes_after_cached_chunks_without_translating_them(monkeypatch) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    monkeypatch.setattr(translation, "_load_model", lambda: (model, tokenizer))

    result = translation.translate_text(
        "A medical report.",
        "hi",
        cached_chunks=["already translated"],
    )

    assert result == "already translated"
    assert model.target_language is None


def test_translation_uses_cuda_when_available(monkeypatch) -> None:
    assert translation._select_device(cuda_available=True) == "cuda"


def test_translation_falls_back_to_cpu_without_cuda(monkeypatch) -> None:
    assert translation._select_device(cuda_available=False) == "cpu"
