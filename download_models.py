import os
from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path("/var/data")
MODELS_DIR = BASE_DIR / "models"
SPAN_DIR = MODELS_DIR / "spans"
TECH_DIR = MODELS_DIR / "tech"

SPAN_REPO = os.environ["HF_SPAN_REPO"]
TECH_REPO = os.environ["HF_TECH_REPO"]
HF_TOKEN = os.environ["HF_TOKEN"]

SPAN_DIR.mkdir(parents=True, exist_ok=True)
TECH_DIR.mkdir(parents=True, exist_ok=True)

if not (SPAN_DIR / "config.json").exists():
    snapshot_download(
        repo_id=SPAN_REPO,
        repo_type="model",
        local_dir=str(SPAN_DIR),
        token=HF_TOKEN,
        local_dir_use_symlinks=False,
    )

if not (TECH_DIR / "best_model.pt").exists():
    snapshot_download(
        repo_id=TECH_REPO,
        repo_type="model",
        local_dir=str(TECH_DIR),
        token=HF_TOKEN,
        local_dir_use_symlinks=False,
    )

print("Models ready")