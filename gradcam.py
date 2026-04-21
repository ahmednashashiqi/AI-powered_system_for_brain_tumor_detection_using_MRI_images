# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Dict, List
import io, base64, warnings
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ========== إعدادات عامة ==========
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_imagenet_stats():
    try:
        w = EfficientNet_B0_Weights.IMAGENET1K_V1
        meta = getattr(w, "meta", {}) or {}
        mean = meta.get("mean", [0.485, 0.456, 0.406])
        std  = meta.get("std",  [0.229, 0.224, 0.225])
        if not (isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple))):
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    except Exception:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return mean, std

_IMAGENET_MEAN, _IMAGENET_STD = _get_imagenet_stats()

# نفس التطبيع المستخدم بالتدريب
_TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

def _build_hooked_model(num_classes: int = 4) -> nn.Module:
    """
    يبني EfficientNet-B0 ويبدّل الرأس لعدد الفئات.
    """
    m = efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m

def _load_state(m: nn.Module, path: str, device: torch.device) -> None:
    """
    تحميل وزن checkpoint مع إزالة 'module.' إن وجدت.
    """
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif not isinstance(sd, dict):
        # ربما حُفظ الموديل كاملًا
        m.load_state_dict(sd.state_dict())
        return

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_sd[k] = v
    m.load_state_dict(new_sd, strict=False)

    # تحقق سريع من تطابق عدد الرؤوس
    out_feats = m.classifier[1].weight.shape[0]
    if out_feats != m.classifier[1].out_features:
        raise RuntimeError(f"Mismatch in head shape: {out_feats} vs {m.classifier[1].out_features}")

def _colormap_on_image(img_np: np.ndarray, cam: np.ndarray, alpha: float=0.45) -> Image.Image:
    """
    يركّب خريطة ألوان على الصورة الأصلية.
    img_np: HxWx3 [0..255], cam: HxW [0..1]
    """
    import matplotlib.cm as cm
    cam = np.clip(cam, 0.0, 1.0)
    heatmap = (cm.jet(cam)[..., :3] * 255.0).astype(np.uint8)  # HxWx3
    heatmap = Image.fromarray(heatmap).resize((img_np.shape[1], img_np.shape[0]), Image.BICUBIC)
    heatmap = np.array(heatmap)
    overlay = (alpha*heatmap + (1.0 - alpha)*img_np).astype(np.uint8)
    return Image.fromarray(overlay)

def _pick_target_layer(model: nn.Module):
    """
    يختار آخر طبقة feature مناسبة لعمل Grad-CAM.
    في EfficientNet-B0 غالبًا model.features[-1] كافي،
    لكن نأخذ آخر Conv2d لو حبّينا نكون أكثر أمانًا.
    """
    # جرّب آخر Conv2d
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv if last_conv is not None else model.features[-1]

def make_gradcam_on_pil(pil_img: Image.Image,
                        ckpt_path: str = "outputs/best_model.pth",
                        num_classes: int = 4,
                        device: Optional[str] = None) -> str:
    """
    يحسب Grad-CAM ويرجع Base64 (PNG) لتراكب الهِيت-ماب على الصورة.
    لو صار أي خطأ، يرفع استثناء واضح (حتى يظهر في واجهة Flask عندك).
    """
    d = torch.device(device or _DEVICE)
    model = _build_hooked_model(num_classes=num_classes).to(d).eval()
    _load_state(model, ckpt_path, d)

    target_layer = _pick_target_layer(model)

    activations = []
    gradients = []

    # hook للاحتفاظ بـ activations
    def fwd_hook(_, __, output):
        activations.clear()
        activations.append(output)

    # بديل موثوق: نسجّل hook مباشرةً على الـTensor الناتج من forward (أسفل) عبر register_hook
    # لكن نحتاج أولاً التقاط الـoutput عبر forward hook، ثم على نفس الـTensor نضيف hook للـgrad.

    h_fwd = target_layer.register_forward_hook(fwd_hook)

    try:
        # الإدخال
        x = _TFMS(pil_img).unsqueeze(0).to(d)
        x.requires_grad_(True)

        # مرّة للأمام حتى نلتقط activations
        logits = model(x)
        if not activations:
            raise RuntimeError("Failed to capture activations. Target layer may be wrong.")

        # سجّل hook للـgrad على الـTensor نفسه
        activations[0].retain_grad()
        def _tensor_grad_hook(grad):
            gradients.clear()
            gradients.append(grad)
            return grad
        activations[0].register_hook(_tensor_grad_hook)

        # التنبؤ الأعلى
        pred_idx = int(logits.argmax(dim=1).item())
        score = logits[0, pred_idx]  # scalar حقيقي

        # backward للحصول على gradients
        model.zero_grad(set_to_none=True)
        with torch.autograd.set_detect_anomaly(False):
            score.backward()

        if not gradients:
            raise RuntimeError("Failed to capture gradients. Check hooks or AMP mode.")

        # activations: [1, C, H, W], gradients: [1, C, H, W]
        acts = activations[0].detach().cpu()[0]   # C,H,W
        grads = gradients[0].detach().cpu()[0]    # C,H,W

        # الأوزان = متوسط gradients على H,W
        weights = grads.mean(dim=(1, 2))          # C
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))  # H,W

        # تطبيع [0..1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        cam_np = cam.numpy()

        # إنتاج صورة overlay وإرجاعها Base64
        img_np = np.array(pil_img.convert("RGB"))
        over = _colormap_on_image(img_np, cam_np, alpha=0.45)
        buf = io.BytesIO()
        over.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    finally:
        # إزالة الـ hooks
        try:
            h_fwd.remove()
        except Exception:
            pass
