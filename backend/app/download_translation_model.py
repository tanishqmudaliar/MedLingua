import argparse
import logging
import os
from pathlib import Path
import sys
import types

from .config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_downloader")

INDICTRANS2_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"
M2M100_MODEL = "facebook/m2m100_418M"


def apply_transformers_compatibility_shims() -> None:
    """Applies compatibility shims for IndicTrans2 with modern transformers (v5+)."""
    import transformers
    import transformers.tokenization_utils
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    # 1. Compatibility shim for IndicTransToolkit with modern Transformers
    if not hasattr(transformers.tokenization_utils, "PreTrainedTokenizerBase"):
        transformers.tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase

    # 2. Tokenizer special tokens map initialization shim
    if not hasattr(PreTrainedTokenizerBase, "_patched_for_indictrans"):
        _orig_new = PreTrainedTokenizerBase.__new__

        def _patched_new(cls, *args, **kwargs):
            instance = _orig_new(cls)
            instance._special_tokens_map = dict.fromkeys(instance.SPECIAL_TOKENS_ATTRIBUTES)
            return instance

        PreTrainedTokenizerBase.__new__ = _patched_new
        PreTrainedTokenizerBase._patched_for_indictrans = True

    # 3. Compatibility shim for IndicTrans2 configuration importing deprecated transformers.onnx
    if "transformers.onnx" not in sys.modules:
        onnx_mod = types.ModuleType("transformers.onnx")
        onnx_utils = types.ModuleType("transformers.onnx.utils")
        onnx_mod.OnnxConfig = type("OnnxConfig", (), {})
        onnx_mod.OnnxSeq2SeqConfigWithPast = type("OnnxSeq2SeqConfigWithPast", (onnx_mod.OnnxConfig,), {})
        onnx_utils.compute_effective_axis_dimension = lambda *a, **k: 1
        onnx_mod.utils = onnx_utils
        sys.modules["transformers.onnx"] = onnx_mod
        sys.modules["transformers.onnx.utils"] = onnx_utils
        transformers.onnx = onnx_mod

    # 4. Compatibility shim for PreTrainedModel._tie_or_clone_weights in transformers v5+
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "_tie_or_clone_weights"):
        def _compat_tie_or_clone(self, output_embeddings, input_embeddings):
            output_embeddings.weight = input_embeddings.weight
        PreTrainedModel._tie_or_clone_weights = _compat_tie_or_clone

    # 5. Compatibility shim for IndicTransForConditionalGeneration.tie_weights keyword arguments
    import transformers.dynamic_module_utils as dmu
    if not getattr(dmu, "_patched_for_indictrans", False):
        _orig_get_class = dmu.get_class_from_dynamic_module
        def _patched_get_class(*args, **kwargs):
            cls = _orig_get_class(*args, **kwargs)
            if hasattr(cls, "tie_weights"):
                _orig_tie = cls.tie_weights
                def _compat_tie(self, *a, **kw):
                    return _orig_tie(self)
                cls.tie_weights = _compat_tie
            return cls
        dmu.get_class_from_dynamic_module = _patched_get_class
        dmu._patched_for_indictrans = True


def get_cache_info(repo_id: str) -> dict:
    """Calculates disk size and file list of a cached repo in huggingface/hub."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_folder_name = "models--" + repo_id.replace("/", "--")
    repo_path = cache_dir / repo_folder_name
    if not repo_path.exists():
        return {"cached": False, "size_mb": 0.0, "has_weights": False, "files": []}

    files = [f for f in repo_path.glob("**/*") if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)
    size_mb = total_size / (1024 * 1024)
    has_weights = any(
        f.name.endswith((".bin", ".safetensors", ".pt")) and f.stat().st_size > 50 * 1024 * 1024
        for f in files
    )
    return {
        "cached": True,
        "size_mb": round(size_mb, 2),
        "has_weights": has_weights,
        "files": [f.name for f in files if f.name.endswith((".bin", ".safetensors", ".json", ".py", ".model"))],
    }


def download_indictrans2(token: str | None = None) -> bool:
    """Attempts to download and cache IndicTrans2."""
    apply_transformers_compatibility_shims()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"\n[DOWNLOAD] Attempting download for {INDICTRANS2_MODEL}...")
    try:
        print("  -> Downloading/verifying tokenizer...")
        AutoTokenizer.from_pretrained(INDICTRANS2_MODEL, trust_remote_code=True, token=token)
        print("  -> Downloading/verifying model weights (approx 1.03 GB)...")
        AutoModelForSeq2SeqLM.from_pretrained(INDICTRANS2_MODEL, trust_remote_code=True, token=token)
        print(f"  [SUCCESS] {INDICTRANS2_MODEL} successfully downloaded and cached!")
        return True
    except Exception as exc:
        print(f"  [ERROR] IndicTrans2 download failed: {exc}")
        print("  [HINT] IndicTrans2 is a gated repository on Hugging Face.")
        print(f"  [HINT] Ensure you accepted license terms at: https://huggingface.co/{INDICTRANS2_MODEL}")
        print("  [HINT] And check that HF_TOKEN is correctly set in backend/.env")
        return False


def download_m2m100() -> bool:
    """Attempts to download and cache M2M100 fallback model."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"\n[DOWNLOAD] Checking/downloading {M2M100_MODEL} (offline fallback)...")
    try:
        AutoTokenizer.from_pretrained(M2M100_MODEL)
        AutoModelForSeq2SeqLM.from_pretrained(M2M100_MODEL)
        print(f"  [SUCCESS] {M2M100_MODEL} is cached and ready!")
        return True
    except Exception as exc:
        print(f"  [ERROR] M2M100 download failed: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="MedLingua Translation Model Downloader & Verification")
    parser.add_argument(
        "--model",
        choices=["indictrans2", "m2m100", "all", "check"],
        default="all",
        help="Model to download/verify (default: all)",
    )
    parser.add_argument("--check-only", action="store_true", help="Only check cache status without downloading")
    args = parser.parse_args()

    settings = get_settings()

    print("=" * 70)
    print("       MedLingua Neural Translation Model Cache & Downloader")
    print("=" * 70)

    indic_info = get_cache_info(INDICTRANS2_MODEL)
    m2m_info = get_cache_info(M2M100_MODEL)

    print(f"\n1. {INDICTRANS2_MODEL} (Primary Indic Model):")
    print(f"   - Cached: {indic_info['cached']} ({indic_info['size_mb']} MB)")
    print(f"   - Weights present: {indic_info['has_weights']}")

    print(f"\n2. {M2M100_MODEL} (Offline Fallback Model):")
    print(f"   - Cached: {m2m_info['cached']} ({m2m_info['size_mb']} MB)")
    print(f"   - Weights present: {m2m_info['has_weights']}")

    if args.check_only or args.model == "check":
        print("\n[CHECK COMPLETED] Model cache inspection finished.")
        return

    # Check HF Token
    hf_token = settings.hf_token
    token_display = f"{hf_token[:6]}...{hf_token[-4:]}" if hf_token and len(hf_token) > 10 else "Not set"
    print(f"\nConfigured HF_TOKEN: {token_display}")

    # Action execution
    if args.model in ("indictrans2", "all"):
        if not indic_info["has_weights"]:
            print(f"\nIndicTrans2 weights not detected. Initiating download...")
            download_indictrans2(hf_token)
        else:
            print(f"\nIndicTrans2 weights already present in cache.")

    if args.model in ("m2m100", "all"):
        if not m2m_info["has_weights"]:
            print(f"\nM2M100 fallback weights not detected. Initiating download...")
            download_m2m100()
        else:
            print(f"\nM2M100 fallback weights already present in cache ({m2m_info['size_mb']} MB).")

    # Final summary
    indic_after = get_cache_info(INDICTRANS2_MODEL)
    m2m_after = get_cache_info(M2M100_MODEL)
    print("\n" + "=" * 70)
    print("Status Summary:")
    print(f" - IndicTrans2 ready: {indic_after['has_weights']}")
    print(f" - M2M100 fallback ready: {m2m_after['has_weights']}")
    if indic_after["has_weights"] or m2m_after["has_weights"]:
        print("At least one translation engine is fully ready for clinical translation.")
    else:
        print("[WARNING] No translation weights found. Translation will fall back to extractive summary.")
    print("=" * 70)


if __name__ == "__main__":
    main()
