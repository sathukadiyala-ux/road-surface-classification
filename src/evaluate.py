"""Evaluate best model on the test set: classification report + confusion matrix."""
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from config import *
from data import get_loaders
from model import build_model

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    _, _, te, classes = get_loaders()
    ck = torch.load(MODEL_DIR / "best.pt", map_location=dev)
    model = build_model(len(ck["classes"]), pretrained=False).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in te:
            ps += model(x.to(dev)).argmax(1).cpu().tolist(); ys += y.tolist()
    rep = classification_report(ys, ps, target_names=classes, digits=3)
    print(rep)
    OUT_DIR.mkdir(exist_ok=True)
    open(OUT_DIR / "classification_report.txt", "w").write(rep)
    cm = confusion_matrix(ys, ps)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
    plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    print("Saved outputs/confusion_matrix.png and classification_report.txt")


if __name__ == "__main__":
    main()
