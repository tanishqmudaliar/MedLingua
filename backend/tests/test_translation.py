import pytest

from app import translation
from app.translation import (
    INDICTRANS2_TAGS,
    LANGUAGES,
    TranslationModelUnavailable,
    _select_device,
    translate_summary,
    translate_text,
)


def test_translation_languages_cover_requested_targets() -> None:
    assert LANGUAGES == {"hi": "Hindi", "mr": "Marathi", "ta": "Tamil"}
    assert INDICTRANS2_TAGS == {
        "hi": "hin_Deva",
        "mr": "mar_Deva",
        "ta": "tam_Taml",
    }


def test_translation_device_selection() -> None:
    assert _select_device(cuda_available=True) == "cuda"
    assert _select_device(cuda_available=False) == "cpu"


def test_translate_summary_unsupported_language() -> None:
    with pytest.raises(ValueError, match="Unsupported translation language"):
        translate_summary("Clinical summary text", "de")  # type: ignore


def test_translate_summary_empty_text() -> None:
    assert translate_summary("", "hi") == ""
    assert translate_summary("   \n\n  ", "mr") == ""


def test_translate_summary_mocked(monkeypatch) -> None:
    class MockModel:
        def generate(self, **kwargs):
            return [[101, 102]]

    class MockTokenizer:
        def __call__(self, batch, **kwargs):
            class Inputs(dict):
                def to(self, device):
                    return self
            return Inputs({"input_ids": [101]})

        def batch_decode(self, outputs, skip_special_tokens=True):
            return ["क्लिनिकल सारांश अनुवाद"]

    class MockProcessor:
        def preprocess_batch(self, sentences, src_lang, tgt_lang):
            return [f"processed_{s}" for s in sentences]

        def postprocess_batch(self, decoded, lang):
            return decoded

    monkeypatch.setattr(
        translation,
        "_load_indictrans2",
        lambda: (MockModel(), MockTokenizer(), MockProcessor()),
    )

    result = translate_summary("Patient has mild fever. Follow up in 3 days.", "hi")
    assert "क्लिनिकल सारांश अनुवाद" in result


def test_translate_text_backward_compatibility(monkeypatch) -> None:
    monkeypatch.setattr(
        translation,
        "translate_summary",
        lambda text, lang: f"Translated ({lang}): {text}",
    )
    progress_calls = []

    def on_progress(done, total, text):
        progress_calls.append((done, total, text))

    res = translate_text("Normal chest radiograph.", "mr", on_progress=on_progress)
    assert res == "Translated (mr): Normal chest radiograph."
    assert len(progress_calls) == 2
    assert progress_calls[0] == (0, 1, "")
    assert progress_calls[1] == (1, 1, "Translated (mr): Normal chest radiograph.")
