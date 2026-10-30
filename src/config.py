"""Central configuration for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # put your downloaded images here: raw/<class_name>/*.jpg
SPLIT_DIR = DATA_DIR / "split"      # created by prepare_dataset.py
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

# Road surface classes (folder names under data/raw/). Edit to match your dataset.
CLASSES = ["dry", "wet", "snow", "gravel", "damaged"]

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 5        # epochs training only the new classifier head
EPOCHS_FINETUNE = 10   # epochs fine-tuning the whole network
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
SEED = 42

# Finite automaton settings
CONFIDENCE_THRESHOLD = 0.60   # below this the CNN output becomes the UNKNOWN symbol
CONFIRM_FRAMES = 5            # consecutive frames needed to change road state
