"""
Split data/raw/<class>/*.jpg into data/split/{train,val,test}/<class>/.

Usage:
    python src/prepare_dataset.py                # 70/15/15 split
    python src/prepare_dataset.py --synthetic    # generate a synthetic demo dataset first
"""
import argparse, random, shutil
from pathlib import Path
from config import RAW_DIR, SPLIT_DIR, CLASSES, SEED

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def make_synthetic(n_per_class=120, size=224):
    """Procedural road textures so the whole pipeline can be tested without a dataset.
    NOT a substitute for real data -- replace with a real road dataset for results."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    rng = np.random.default_rng(SEED)
    for c in CLASSES:
        d = RAW_DIR / c
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            base = rng.integers(70, 110)
            img = np.full((size, size, 3), base, dtype=np.float32)
            img += rng.normal(0, 8, img.shape)
            if c == "wet":
                img *= 0.65
                for y in rng.integers(0, size, 3):
                    img[max(0, y-6):y+6, :, :] += 70
            elif c == "snow":
                img = img * 0.3 + 200 + rng.normal(0, 10, img.shape)
            elif c == "gravel":
                img += rng.normal(0, 35, img.shape)
                img[..., 0] += 20
            im = Image.fromarray(np.clip(img, 0, 255).astype("uint8"))
            dr = ImageDraw.Draw(im)
            if c == "damaged":
                for _ in range(int(rng.integers(2, 5))):
                    x, y = (int(v) for v in rng.integers(20, size-20, 2))
                    r = int(rng.integers(10, 30))
                    dr.ellipse([x-r, y-r, x+r, y+r], fill=(25, 25, 25))
                x0 = int(rng.integers(0, size))
                dr.line([x0, 0, x0 + int(rng.integers(-40, 40)), size], fill=(15, 15, 15), width=3)
            im = im.filter(ImageFilter.GaussianBlur(0.8))
            im.save(d / f"{c}_{i:04d}.jpg")
    print(f"Synthetic dataset written to {RAW_DIR}")


def split(train=0.70, val=0.15):
    random.seed(SEED)
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    for c in CLASSES:
        files = sorted(p for p in (RAW_DIR / c).glob("*") if p.suffix.lower() in EXTS)
        if not files:
            raise SystemExit(f"No images found in {RAW_DIR / c}")
        random.shuffle(files)
        n = len(files); a = int(n * train); b = int(n * (train + val))
        parts = {"train": files[:a], "val": files[a:b], "test": files[b:]}
        for name, fl in parts.items():
            out = SPLIT_DIR / name / c
            out.mkdir(parents=True, exist_ok=True)
            for f in fl:
                shutil.copy2(f, out / f.name)
        print(f"{c:10s} total={n:4d} train={len(parts['train'])} val={len(parts['val'])} test={len(parts['test'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.synthetic:
        make_synthetic()
    split()
