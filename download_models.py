"""
download_models.py
===================
Download required Hugging Face models and tokenizers ahead of time to a
local cache directory, so subsequent scripts can load them from disk.

Usage:
    python download_models.py --config configs/config.yaml
    python download_models.py --config configs/config.yaml --cache-dir model_cache --local-only
"""

import argparse
import json
from pathlib import Path

import yaml
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

MODEL_CLASS_MAP = {
    "nli_model": AutoModelForSequenceClassification,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models and tokenizers to a local cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Local directory to cache downloaded models. Overrides config path.",
    )
    parser.add_argument(
        "--model-keys",
        nargs="+",
        default=["deberta_base", "nli_model"],
        help="Model config keys to download from the config file.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only verify that models already exist locally; do not download from the internet.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def download_model(model_name: str, model_key: str, cache_dir: Path, local_only: bool) -> None:
    print(f"Downloading tokenizer for {model_key}: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        local_files_only=local_only,
    )
    print(f"  Tokenizer ready: {tokenizer.__class__.__name__}")

    model_cls = MODEL_CLASS_MAP.get(model_key, AutoModel)
    print(f"Downloading model for {model_key}: {model_name} ({model_cls.__name__})")
    model_cls.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        local_files_only=local_only,
    )
    print(f"  Model ready in cache: {model_name}\n")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = project_root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    paths_cfg = config.get("paths", {})
    cache_dir = Path(args.cache_dir) if args.cache_dir else project_root / Path(paths_cfg.get("model_cache_dir", "model_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    models_cfg = config.get("models", {})
    if not models_cfg:
        raise ValueError("No models found in config under 'models'.")

    print(f"Using local model cache: {cache_dir}")
    print(f"Local-only mode: {args.local_only}\n")

    downloaded = []
    for model_key in args.model_keys:
        if model_key not in models_cfg:
            print(f"Warning: model key '{model_key}' not found in config; skipping.")
            continue
        model_name = models_cfg[model_key]
        download_model(model_name, model_key, cache_dir, args.local_only)
        downloaded.append(model_key)

    print("Download finished.")
    print(json.dumps({"cache_dir": str(cache_dir), "models": downloaded}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
