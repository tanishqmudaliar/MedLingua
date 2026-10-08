from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from .translation import MODEL_NAME


def main() -> None:
    print(f"Downloading {MODEL_NAME} for local Hindi, Marathi, and Tamil translation...")
    M2M100Tokenizer.from_pretrained(MODEL_NAME)
    M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
    print("Translation model is ready. Translations run locally at runtime.")


if __name__ == "__main__":
    main()
