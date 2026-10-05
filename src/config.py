from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

PHYSIONET_ROOT = Path(os.getenv("PHYSIONET_ROOT"))

TRAIN_A = PHYSIONET_ROOT / "training_setA"
TRAIN_B = PHYSIONET_ROOT / "training_setB"

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)