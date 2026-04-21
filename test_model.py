# -*- coding: utf-8 -*-
"""
Test the trained classifier on the Test set and print detailed metrics:
Precision, Recall, F1-Score, Support, Accuracy

Usage:
    # اختبار على dataset/Testing (الافتراضي):
    python test_model.py

    # اختبار على مجلد مخصص:
    python test_model.py --dir test
    python test_model.py --dir path/to/my/images

Outputs:
    - Console: Full classification report
    - reports/test_metrics.txt: Detailed metrics
    - reports/test_confusion_matrix.png: Confusion matrix
"""

import argparse
import os, json, platform
from pathlib import Path

# لضمان حفظ الصور بدون فتح نافذة
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
except ModuleNotFoundError:
    raise SystemExit(
        "❌ scikit-learn غير مُثبّت.\n"
        "ثبّته: pip install scikit-learn\n"
    )

# ====== Paths ======
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
RPT_DIR = ROOT / "reports"

CKPT_PATH = OUT_DIR / "best_model.pth"
CLASSES_JSON = OUT_DIR / "classes.json"

RPT_DIR.mkdir(parents=True, exist_ok=True)

# ====== Settings ======
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== تحميل الأصناف ======
def load_classes():
    if not CLASSES_JSON.exists():
        raise FileNotFoundError(f"classes.json not found: {CLASSES_JSON}\nRun train_model.py first.")
    raw = json.loads(CLASSES_JSON.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    elif isinstance(raw, dict):
        keys = sorted([int(k) for k in raw.keys()])
        return [raw[str(k)] for k in keys]
    raise ValueError("Invalid classes.json format")

# ====== Dataset مخصص يتجاهل المجلدات الفارغة ======
class CustomImageFolder(Dataset):
    """Dataset مخصص يقرأ فقط المجلدات التي تحتوي على صور."""
    def __init__(self, root_dir, transform=None, valid_extensions=None):
        if valid_extensions is None:
            valid_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
        
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        # البحث عن المجلدات التي تحتوي على صور
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            images = [f for f in class_dir.iterdir() 
                     if f.is_file() and f.suffix.lower() in valid_extensions]
            
            if len(images) > 0:  # فقط المجلدات التي تحتوي على صور
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.classes)
                    self.classes.append(class_name)
                
                for img_path in images:
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading image {img_path}: {e}")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ====== تحويلات التقييم ======
def make_eval_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

# ====== بناء النموذج ======
def build_model(num_classes: int):
    model = models.efficientnet_b0(weights=None)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model

# ====== تحميل الأوزان ======
def load_checkpoint(model: nn.Module, ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nRun train_model.py first.")
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        model.load_state_dict(ckpt.state_dict())
        return
    # إزالة "module." إن وُجدت
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)

# ====== تقييم على Test set ======
@torch.no_grad()
def evaluate_test(model, test_loader, test_classes, all_classes):
    """
    تقييم النموذج على test set مع معالجة عدم تطابق الفئات.
    
    Args:
        model: النموذج المدرب
        test_loader: DataLoader للاختبار
        test_classes: قائمة بالفئات الموجودة في test set
        all_classes: قائمة بجميع فئات النموذج من classes.json
    
    Returns:
        preds: التوقعات (indices في test_classes)
        labels: الملصقات الحقيقية (indices في test_classes)
    """
    model.eval()
    all_pred, all_true = [], []
    
    # إنشاء mapping من test classes إلى model classes
    all_class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    test_class_to_model_idx = {}
    for test_class in test_classes:
        if test_class in all_class_to_idx:
            test_class_to_model_idx[test_class] = all_class_to_idx[test_class]
        else:
            raise ValueError(f"Test class '{test_class}' not found in model classes")
    
    # إنشاء mapping عكسي: model index -> test index
    model_idx_to_test_idx = {model_idx: test_idx 
                             for test_idx, (_, model_idx) in enumerate(test_class_to_model_idx.items())}
    
    for x, y in test_loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            logits = model(x)
        
        # التوقعات من النموذج (indices في all_classes)
        pred_model = logits.argmax(1).cpu()
        
        # تحويل التوقعات إلى test indices (فقط للفئات الموجودة)
        pred_test = []
        for p in pred_model:
            if p.item() in model_idx_to_test_idx:
                pred_test.append(model_idx_to_test_idx[p.item()])
            else:
                # إذا كانت التوقعة لفئة غير موجودة في الاختبار، نستخدم -1 (سيتم تجاهلها لاحقاً)
                pred_test.append(-1)
        
        # الملصقات الحقيقية (indices في test_classes)
        labels_test = y.cpu().tolist()
        
        all_pred.extend(pred_test)
        all_true.extend(labels_test)
    
    # إزالة التوقعات غير الصالحة
    preds = np.array(all_pred)
    labels = np.array(all_true)
    valid_mask = preds >= 0
    preds = preds[valid_mask]
    labels = labels[valid_mask]
    
    return preds, labels

# ====== رسم Confusion Matrix ======
def plot_confusion_matrix(cm: np.ndarray, class_names, save_path: Path, show_percentages=True):
    """
    رسم Confusion Matrix مع تحسينات بصرية.
    
    Args:
        cm: مصفوفة الارتباك
        class_names: أسماء الفئات
        save_path: مسار حفظ الصورة
        show_percentages: عرض النسب المئوية بجانب الأرقام
    """
    # حساب النسب المئوية
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # استخدام خريطة ألوان أفضل
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, aspect='auto', vmin=0, vmax=cm.max())
    ax.figure.colorbar(im, ax=ax, label='عدد الصور', fraction=0.046, pad=0.04)
    
    ax.set_title("Confusion Matrix — Test Set", fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_ylabel("True Label (الحقيقة)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label (التوقع)", fontsize=13, fontweight="bold")
    
    # إضافة خطوط فاصلة
    ax.set_xticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    # إضافة الأرقام والنسب المئوية داخل الخلايا
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            if show_percentages and cm.sum(axis=1)[i] > 0:
                percent = f"{cm_percent[i, j]:.1f}%"
                text = f"{count}\n({percent})"
            else:
                text = str(count)
            
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10, fontweight="bold",
                    linespacing=1.2)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    
    # فتح الصورة تلقائياً (على Windows)
    if platform.system().lower() == "windows":
        try:
            os.startfile(str(save_path))
        except Exception:
            pass  # تجاهل الأخطاء إذا فشل فتح الصورة

# ====== Main ======
def main():
    parser = argparse.ArgumentParser(description="Test classifier on images")
    parser.add_argument("--dir", type=str, default=None, 
                        help="Path to test images directory (default: dataset/Testing)")
    args = parser.parse_args()
    
    # تحديد مجلد الاختبار
    if args.dir:
        TEST_DIR = Path(args.dir)
    else:
        TEST_DIR = ROOT / "dataset" / "Testing"
    
    print("=" * 60)
    print("Brain Tumor Classifier — Test Set Evaluation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # تحميل الأصناف من classes.json
    all_classes = load_classes()
    num_classes = len(all_classes)
    print(f"Model classes ({num_classes}): {', '.join(all_classes)}")
    print()

    # التحقق من Test set
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")
    
    # قراءة الفئات الموجودة فعلياً في مجلد الاختبار (تجاهل المجلدات الفارغة)
    test_ds = CustomImageFolder(str(TEST_DIR), transform=make_eval_transforms())
    test_classes = test_ds.classes  # الفئات الموجودة في Testing/
    
    if len(test_ds) == 0:
        raise ValueError(f"No images found in {TEST_DIR}")
    
    print(f"Test set: {len(test_ds)} images")
    print(f"Test classes found: {', '.join(test_classes)}")
    
    # التحقق من تطابق الفئات
    missing_in_test = set(all_classes) - set(test_classes)
    if missing_in_test:
        print(f"⚠️  Warning: Some model classes not found in test set: {', '.join(missing_in_test)}")
        print("   Evaluation will only include classes present in test set.")
    print()

    # DataLoader
    if platform.system().lower().startswith("win"):
        num_workers = 0
    else:
        num_workers = 4
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    # بناء النموذج وتحميل الأوزان
    print("Loading model...")
    model = build_model(num_classes=num_classes).to(DEVICE).eval()
    load_checkpoint(model, CKPT_PATH)
    print("✓ Model loaded")
    print()

    # التقييم
    print("Evaluating on test set...")
    preds, labels = evaluate_test(model, test_loader, test_classes, all_classes)
    print("✓ Evaluation complete")
    print()

    # حساب المقاييس
    accuracy = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=range(len(test_classes)))
    report = classification_report(labels, preds, target_names=test_classes, digits=4, zero_division=0)

    # طباعة النتائج
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    print("Classification Report:")
    print(report)
    print()
    print("Confusion Matrix:")
    print(cm)
    print()

    # حفظ النتائج
    report_file = RPT_DIR / "test_metrics.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Brain Tumor Classifier — Test Set Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {CKPT_PATH}\n")
        f.write(f"Test set: {TEST_DIR}\n")
        f.write(f"Total test images: {len(test_ds)}\n")
        f.write(f"Model classes: {', '.join(all_classes)}\n")
        f.write(f"Test classes found: {', '.join(test_classes)}\n")
        f.write(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n")

    cm_file = RPT_DIR / "test_confusion_matrix.png"
    plot_confusion_matrix(cm, test_classes, cm_file)

    print("=" * 60)
    print("Outputs saved:")
    print(f"  • Metrics: {report_file}")
    print(f"  • Confusion Matrix: {cm_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
