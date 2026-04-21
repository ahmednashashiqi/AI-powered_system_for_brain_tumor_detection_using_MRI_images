# -*- coding: utf-8 -*-
"""
Train a brain-tumor classifier (EfficientNet-B0) on your Kaggle dataset.

Expected layout:
dataset/
  Training/
    glioma/ | meningioma/ | pituitary/ | notumor/
  Testing/
    glioma/ | meningioma/ | pituitary/ | notumor/

Outputs:  outputs/best_model.pth, outputs/classes.json
Reports:  reports/metrics.txt, reports/confusion_matrix.png
"""

import os, sys, json, time, platform
from pathlib import Path

# ====== لضمان حفظ الصور بدون فتح نافذة ======
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ====== استيراد آمن لـ scikit-learn مع رسالة مفهومة ======
try:
    from sklearn.metrics import classification_report, confusion_matrix
except ModuleNotFoundError:
    raise SystemExit(
        "❌ scikit-learn غير مُثبّت داخل البيئة.\n"
        "ثبّته بالأمر:\n\n"
        "    pip install scikit-learn numpy matplotlib\n\n"
        "ثم أعد التشغيل:\n"
        "    python train_model.py"
    )

# ====== CUDA tweaks (GPU أفضل) ======
import torch.backends.cudnn as cudnn
torch.set_float32_matmul_precision("high")
cudnn.benchmark = True

# ====== Paths ======
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR  = DATA_DIR / "Testing"
OUT_DIR   = ROOT / "outputs"
RPT_DIR   = ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RPT_DIR.mkdir(parents=True, exist_ok=True)

# ====== Hyperparams ======
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 1e-4
VAL_SPLIT  = 0.2  # نسبة من training تصبح validation
SEED       = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

def check_dataset_layout():
    """يتأكد من وجود مجلدات الداتا بالشكل المتوقع ويطبع ملاحظة واضحة."""
    expected = [
        TRAIN_DIR / "glioma",
        TRAIN_DIR / "meningioma",
        TRAIN_DIR / "pituitary",
        TRAIN_DIR / "notumor",
        TEST_DIR / "glioma",
        TEST_DIR / "meningioma",
        TEST_DIR / "pituitary",
        TEST_DIR / "notumor",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        msg = "❌ المجلدات التالية غير موجودة:\n- " + "\n- ".join(missing)
        msg += (
            "\n\nتأكد أن المسار كالتالي:\n"
            "dataset/Training/{glioma, meningioma, pituitary, notumor}\n"
            "dataset/Testing/{glioma, meningioma, pituitary, notumor}\n"
        )
        raise SystemExit(msg)

def make_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    tf_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tf_train, tf_eval

def get_loaders():
    tf_train, tf_eval = make_transforms()
    full_train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=tf_train)
    test_ds       = datasets.ImageFolder(str(TEST_DIR),  transform=tf_eval)

    n_total = len(full_train_ds)
    n_val   = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    # DataLoaders: على ويندوز من الأفضل جعل num_workers=0 لتفادي مشاكل fork
    if platform.system().lower().startswith("win"):
        num_workers = 0
        persistent = False
    else:
        num_workers = 4
        persistent = True

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=persistent
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=persistent
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=persistent
    )

    classes = full_train_ds.classes
    return train_loader, val_loader, test_loader, classes

def build_model(num_classes: int):
    # تحميل أوزان ImageNet الافتراضية
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optim_):
    """تدريب آمن FP16 على GPU باستخدام torch.amp"""
    model.train()
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    loss_sum, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optim_.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            logits = model(x)
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optim_)
        scaler.update()

        loss_sum += loss.item() * x.size(0)
        pred = logits.detach().argmax(1)
        correct += (pred == y).sum().item()
        total   += x.size(0)

    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_pred, all_true = [], []
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            logits = model(x)
            loss   = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total   += x.size(0)
        all_pred.extend(pred.cpu().tolist())
        all_true.extend(y.cpu().tolist())
    return loss_sum/total, correct/total, np.array(all_pred), np.array(all_true)

def plot_confusion(cm: np.ndarray, class_names, save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    check_dataset_layout()
    train_loader, val_loader, test_loader, classes = get_loaders()

    # حفظ أسماء الأصناف للواجهة لاحقًا
    (OUT_DIR / "classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model = build_model(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_acc, wait, patience = 0.0, 0, 5
    best_path = OUT_DIR / "best_model.pth"

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        dt = time.time() - t0
        print(f"[{epoch:02d}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes}, best_path)
            wait = 0
            print("  ↳ Saved best model.")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # تقييم نهائي على Test بأفضل وزن
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    print(f"\nTEST: acc={test_acc:.4f} loss={test_loss:.4f}")

    # تقرير سكيت-ليرن
    report_txt = classification_report(labels, preds, target_names=classes, digits=4)
    cm = confusion_matrix(labels, preds)

    (RPT_DIR / "metrics.txt").write_text(
        f"TEST acc={test_acc:.4f} loss={test_loss:.4f}\n\n{report_txt}\n",
        encoding="utf-8"
    )
    plot_confusion(cm, classes, RPT_DIR / "confusion_matrix.png")

    print(f"Saved: {best_path}")
    print(f"Report: {RPT_DIR/'metrics.txt'}")
    print(f"CM img: {RPT_DIR/'confusion_matrix.png'}")

if __name__ == "__main__":
    main()
