from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "waste.db"
HISTORY_DIR = BASE_DIR / "data" / "history"
MODEL_DIR = BASE_DIR / "ai" / "model"
DATASET_DIR = BASE_DIR / "dataset"
DEFAULT_DETECTION_MODEL = "yolo11n.pt"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for directory in (DATABASE_DIR, HISTORY_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)
