# -*- coding: utf-8 -*-
"""
اختبار النموذج على صور خارجية (ليست من Test set)

Usage:
    # إذا الصور في مجلدات منظمة (مثل dataset/External/glioma/, ...):
    python test_external.py --dir dataset/External

    # أو إذا كل الصور في مجلد واحد (بدون labels):
    python test_external.py --dir path/to/images --no-labels

Outputs:
    - Console: نتائج لكل صورة + إحصائيات
    - reports/external_test_results.txt: ملف نصي بالنتائج
"""

import argparse
import json
import platform
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
except ModuleNotFoundError:
    raise SystemExit("❌ scikit-learn غير مُثبّت. ثبّته: pip install scikit-learn")

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
        raise FileNotFoundError(f"classes.json not found: {CLASSES_JSON}")
    raw = json.loads(CLASSES_JSON.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    elif isinstance(raw, dict):
        keys = sorted([int(k) for k in raw.keys()])
        return [raw[str(k)] for k in keys]
    raise ValueError("Invalid classes.json format")

# ====== تحويلات ======
def make_transforms():
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
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        model.load_state_dict(ckpt.state_dict())
        return
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)

# ====== تقييم على صور خارجية ======
@torch.no_grad()
def evaluate_external(model, loader, classes):
    model.eval()
    results = []
    all_pred, all_true = [], []
    
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1)
        
        for i in range(len(preds)):
            pred_idx = int(preds[i].item())
            conf = float(probs[i, pred_idx].item())
            true_idx = int(y[i].item()) if y is not None else None
            all_pred.append(pred_idx)
            if true_idx is not None:
                all_true.append(true_idx)
            results.append({
                "predicted": classes[pred_idx],
                "confidence": conf,
                "true": classes[true_idx] if true_idx is not None else None
            })
    
    return results, np.array(all_pred), np.array(all_true) if all_true else None

# ====== Main ======
def main():
    parser = argparse.ArgumentParser(description="Test model on external images")
    parser.add_argument("--dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--no-labels", action="store_true", help="Images in one folder (no subfolders with class names)")
    args = parser.parse_args()
    
    img_dir = Path(args.dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Directory not found: {img_dir}")
    
    print("=" * 60)
    print("Brain Tumor Classifier — External Images Test")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # تحميل الأصناف
    classes = load_classes()
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {', '.join(classes)}")
    print()
    
    # تحميل الصور
    tfms = make_transforms()
    
    if args.no_labels:
        # كل الصور في مجلد واحد - نستخدم ImageFolder لكن بدون labels حقيقية
        # سنستخدم dataset.ImageFolder لكن سنتجاهل labels في الحساب
        print("Mode: Single folder (no ground truth labels)")
        # نحتاج طريقة أخرى - نقرأ الصور يدوياً
        from torch.utils.data import Dataset
        
        class SingleFolderDataset(Dataset):
            def __init__(self, folder, transform):
                self.folder = Path(folder)
                self.files = sorted([f for f in self.folder.glob("*") 
                                     if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}])
                self.transform = transform
            def __len__(self):
                return len(self.files)
            def __getitem__(self, idx):
                img = Image.open(self.files[idx]).convert("RGB")
                return self.transform(img), -1  # -1 = no label
        
        dataset = SingleFolderDataset(img_dir, tfms)
        file_names = [str(dataset.files[i]) for i in range(len(dataset))]
        has_labels = False
    else:
        # مجلدات منظمة (مثل External/glioma/, External/meningioma/, ...)
        print("Mode: Organized folders (with ground truth labels)")
        dataset = datasets.ImageFolder(str(img_dir), transform=tfms)
        file_names = [str(dataset.samples[i][0]) for i in range(len(dataset))]
        has_labels = True
    
    if len(dataset) == 0:
        raise ValueError(f"No images found in {img_dir}")
    print(f"Images found: {len(dataset)}")
    print()
    
    # DataLoader
    if platform.system().lower().startswith("win"):
        num_workers = 0
    else:
        num_workers = 4
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    # تحميل النموذج
    print("Loading model...")
    model = build_model(num_classes=num_classes).to(DEVICE).eval()
    load_checkpoint(model, CKPT_PATH)
    print("✓ Model loaded")
    print()
    
    # التقييم
    print("Evaluating...")
    results, preds, labels = evaluate_external(model, loader, classes)
    print("✓ Evaluation complete")
    print()
    
    # ربط النتائج بأسماء الملفات
    detailed_results = []
    for i, res in enumerate(results):
        fname = Path(file_names[i]).name
        detailed_results.append({
            "file": fname,
            "predicted": res["predicted"],
            "confidence": res["confidence"],
            "true": res["true"]
        })
    
    # طباعة النتائج
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    
    # توزيع التنبؤات
    pred_counts = {}
    for r in detailed_results:
        p = r["predicted"]
        pred_counts[p] = pred_counts.get(p, 0) + 1
    
    print("Predictions distribution:")
    for cls in classes:
        count = pred_counts.get(cls, 0)
        pct = (count / len(detailed_results)) * 100 if detailed_results else 0
        print(f"  {cls:15s}: {count:4d} ({pct:5.2f}%)")
    print()
    
    # إذا فيه labels حقيقية - حساب Accuracy
    if has_labels and labels is not None and len(labels) > 0:
        accuracy = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)
        report = classification_report(labels, preds, target_names=classes, digits=4, zero_division=0)
        
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print()
        print("Classification Report:")
        print(report)
        print()
        print("Confusion Matrix:")
        print(cm)
        print()
    else:
        print("Note: No ground truth labels available. Showing predictions only.")
        print()
    
    # أول 10 نتائج كمثال
    print("Sample results (first 10):")
    for i, r in enumerate(detailed_results[:10]):
        true_str = f" | True: {r['true']}" if r['true'] else ""
        print(f"  {i+1:2d}. {r['file']:30s} → {r['predicted']:12s} ({r['confidence']:.4f}){true_str}")
    if len(detailed_results) > 10:
        print(f"  ... and {len(detailed_results) - 10} more")
    print()
    
    # حفظ النتائج
    report_file = RPT_DIR / "external_test_results.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Brain Tumor Classifier — External Images Test\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {CKPT_PATH}\n")
        f.write(f"Images directory: {img_dir}\n")
        f.write(f"Total images: {len(detailed_results)}\n")
        f.write(f"Classes: {', '.join(classes)}\n")
        f.write(f"Has labels: {has_labels}\n\n")
        
        if has_labels and labels is not None and len(labels) > 0:
            f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\n")
        
        f.write("Predictions distribution:\n")
        for cls in classes:
            count = pred_counts.get(cls, 0)
            pct = (count / len(detailed_results)) * 100 if detailed_results else 0
            f.write(f"  {cls:15s}: {count:4d} ({pct:5.2f}%)\n")
        f.write("\n\nDetailed results:\n")
        f.write("-" * 60 + "\n")
        for r in detailed_results:
            true_str = f" | True: {r['true']}" if r['true'] else ""
            f.write(f"{r['file']:40s} → {r['predicted']:12s} ({r['confidence']:.4f}){true_str}\n")
    
    print("=" * 60)
    print("Output saved:")
    print(f"  • Results: {report_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
