"""Two-stage training: (1) train new head, (2) fine-tune the full network."""
import json, torch, torch.nn as nn
from tqdm import tqdm
from config import *
from data import get_loaders
from model import build_model, freeze_backbone

torch.manual_seed(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    tot, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            tot += len(y)
    return loss_sum / tot, correct / tot


def main():
    MODEL_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
    tr, va, _, classes = get_loaders()
    model = build_model(len(classes)).to(dev)
    loss_fn = nn.CrossEntropyLoss()
    hist, best = [], 0.0

    stages = [("head", EPOCHS_HEAD, LR_HEAD, True), ("finetune", EPOCHS_FINETUNE, LR_FINETUNE, False)]
    for name, epochs, lr, frozen in stages:
        freeze_backbone(model, frozen)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        for ep in range(epochs):
            tl, ta = run_epoch(model, tr, loss_fn, opt)
            vl, vacc = run_epoch(model, va, loss_fn)
            print(f"[{name}] epoch {ep+1}/{epochs}  train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{vacc:.3f}")
            hist.append(dict(stage=name, epoch=ep+1, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=vacc))
            if vacc >= best:
                best = vacc
                torch.save({"state_dict": model.state_dict(), "classes": classes}, MODEL_DIR / "best.pt")
    json.dump(hist, open(OUT_DIR / "history.json", "w"), indent=2)
    print(f"Best val accuracy: {best:.3f}  -> models/best.pt")


if __name__ == "__main__":
    main()
