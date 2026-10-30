"""
Copy images from a downloaded dataset into data/raw/<class>/ using keywords in the file path.

Usage (from the project folder):
    python src/map_dataset.py C:\path\to\your\dataset

Edit RULES below to match your dataset's folder/file names. First matching rule wins.
"""
import sys, random, shutil
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit("Usage: python src/map_dataset.py <dataset_folder>")

SRC = Path(sys.argv[1])
if not SRC.exists():
    raise SystemExit(f"Folder not found: {SRC}")

DST = Path(__file__).resolve().parent.parent / "data" / "raw"
MAX_PER_CLASS = 1500
random.seed(42)

RULES = [
    ("snow",    ["snow", "ice"]),
    ("gravel",  ["gravel"]),
    ("damaged", ["pothole", "crack", "damage", "severe"]),
    ("wet",     ["wet", "water"]),
    ("dry",     ["dry"]),
]

found = {c: [] for c, _ in RULES}
for f in SRC.rglob("*"):
    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        p = str(f).lower()
        for cls, keys in RULES:
            if any(k in p for k in keys):
                found[cls].append(f)
                break

for cls, files in found.items():
    random.shuffle(files)
    out = DST / cls
    out.mkdir(parents=True, exist_ok=True)
    n = min(len(files), MAX_PER_CLASS)
    for i, f in enumerate(files[:n]):
        shutil.copy2(f, out / f"{cls}_{i:05d}{f.suffix.lower()}")
    print(f"{cls:8s} found={len(files):6d} copied={n}")
