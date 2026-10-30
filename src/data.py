from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import SPLIT_DIR, IMG_SIZE, BATCH_SIZE

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def eval_tf():
    return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def train_tf():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def get_loaders():
    tr = datasets.ImageFolder(SPLIT_DIR / "train", train_tf())
    va = datasets.ImageFolder(SPLIT_DIR / "val", eval_tf())
    te = datasets.ImageFolder(SPLIT_DIR / "test", eval_tf())
    mk = lambda d, s: DataLoader(d, batch_size=BATCH_SIZE, shuffle=s, num_workers=2)
    return mk(tr, True), mk(va, False), mk(te, False), tr.classes
