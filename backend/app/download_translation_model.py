import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .config import get_settings

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    model_name = settings.indictrans_model_name
    print(f"Checking/downloading {model_name} for local Indic translation...")
    try:
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=settings.hf_token)
        AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, token=settings.hf_token)
        print("IndicTrans2 translation model weights cached locally.")
    except Exception as exc:
        print(f"Notice: Translation model download skipped or deferred ({exc}).")
        print("Model will load on-demand when translation jobs execute.")


if __name__ == "__main__":
    main()
