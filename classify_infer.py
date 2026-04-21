# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

# ========== الجهاز ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== المسارات ==========
ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "outputs" / "best_model.pth"
CLASSES_JSON = ROOT / "outputs" / "classes.json"

# ========== أصناف fallback لو classes.json ناقص/فاضي ==========
# ملاحظة: ImageFolder يرتّب المجلدات أبجديًا وقت التدريب. تأكد من التطابق.
FALLBACK_CLASS_ORDER: List[str] = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

# ========== إحضار mean/std المتوافقة مع كل إصدارات torchvision ==========
def _get_imagenet_stats():
    try:
        w = EfficientNet_B0_Weights.IMAGENET1K_V1
        meta = getattr(w, "meta", {}) or {}
        mean = meta.get("mean", [0.485, 0.456, 0.406])
        std  = meta.get("std",  [0.229, 0.224, 0.225])
        # لو رجع None أو شكل غلط، نرجع الافتراضيات
        if not (isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple))):
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    except Exception:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return mean, std

_IMAGENET_MEAN, _IMAGENET_STD = _get_imagenet_stats()

# ========== نفس تحويلات التدريب ==========
IMG_TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# ========== تحميل قائمة الأصناف ==========
def _load_classes() -> Dict[int, str]:
    """
    يحاول قراءة outputs/classes.json.
    يدعم:
      - ["glioma", "meningioma", ...]
      - {"0":"glioma","1":"meningioma",...}
    وإن تعذّر، يستخدم FALLBACK_CLASS_ORDER.
    """
    idx_to_name: List[str] = []
    if CLASSES_JSON.exists():
        try:
            raw = json.loads(CLASSES_JSON.read_text(encoding="utf-8").strip() or "null")
            if isinstance(raw, list) and len(raw) > 0:
                idx_to_name = raw
            elif isinstance(raw, dict) and len(raw) > 0:
                # نجمعها بترتيب المفاتيح الرقمية
                keys = sorted([int(k) for k in raw.keys()])
                idx_to_name = [raw[str(k)] for k in keys]
        except Exception as e:
            print(f"⚠️ classes.json read error: {e}")
    if not idx_to_name:
        print(f"⚠️ Using FALLBACK_CLASS_ORDER: {FALLBACK_CLASS_ORDER}")
        idx_to_name = FALLBACK_CLASS_ORDER
    return {i: name for i, name in enumerate(idx_to_name)}

IDX_TO_NAME = _load_classes()
NUM_CLASSES = len(IDX_TO_NAME)

# ========== بناء النموذج ==========
def _build_model(num_classes: int) -> nn.Module:
    m = efficientnet_b0(weights=None)  # سنحمّل وزننا المدرب لاحقًا
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m

# ========== تحميل الأوزان ==========
def _load_state(model: nn.Module, ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt  # قد يكون state_dict مباشرة
    else:
        # في حالات نادرة يكون كائن model محفوظ بالكامل
        model.load_state_dict(ckpt.state_dict())
        return

    # إزالة بادئة DataParallel "module." إن وُجدت
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v

    model.load_state_dict(new_state, strict=False)

    # تحقق من تطابق عدد الرؤوس مع عدد الأصناف
    head = model.classifier[1].weight
    out_features = head.shape[0]
    if out_features != NUM_CLASSES:
        raise RuntimeError(
            f"Head/out_features mismatch: checkpoint has {out_features} classes "
            f"but IDX_TO_NAME has {NUM_CLASSES}. "
            f"تأكد من صحة outputs/classes.json أو أعد التدريب/التصدير."
        )

# ========== دالة الاستدلال ==========
@torch.no_grad()
def infer_image(pil_img: Image.Image) -> Tuple[str, float, Dict[str, float]]:
    """
    يُرجع: (label_name, confidence, scores_dict_for_all_classes)
    """
    model = _build_model(NUM_CLASSES).to(DEVICE).eval()
    _load_state(model, CKPT_PATH)

    # autocast على GPU فقط
    use_amp = torch.cuda.is_available()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
        x = IMG_TFMS(pil_img).unsqueeze(0).to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    top_idx = int(probs.argmax())
    label = IDX_TO_NAME[top_idx]
    conf = float(probs[top_idx])
    scores = {IDX_TO_NAME[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    return label, conf, scores

# ========== اختبار منفرد ==========
if __name__ == "__main__":
    test_path = ROOT / "images" / "sample_mri.jpg"
    if not test_path.exists():
        raise FileNotFoundError(f"Test image not found: {test_path}")
    img = Image.open(test_path).convert("RGB")
    label, conf, scores = infer_image(img)
    print(label, conf, scores)
