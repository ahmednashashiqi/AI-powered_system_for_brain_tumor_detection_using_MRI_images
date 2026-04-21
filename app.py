# app.py
# -*- coding: utf-8 -*-
# ========= بيئة آمنة بدون TF/Keras =========
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ.setdefault("XFORMERS_DISABLED", "1")

import io, tempfile, subprocess, json, time, datetime, binascii, mimetypes, uuid
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, request, jsonify, render_template, redirect, url_for

import sys, torch, numpy as np
from functools import wraps
from datetime import timedelta

# ---------- إعدادات عامة ----------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# قيود الرفع (سيرفر)
MAX_CONTENT_MB = float(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_CONTENT_LENGTH = int(MAX_CONTENT_MB * 1024 * 1024)  # بايت
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"}
mimetypes.init()

# ---------- أنشئ تطبيق Flask ----------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["JSON_AS_ASCII"] = False
app.config["ENV"] = os.getenv("FLASK_ENV", "production")

# ========== pydicom (اختياري) ==========
try:
    import pydicom
    _HAS_PYDICOM = True
except Exception:
    _HAS_PYDICOM = False

# ========== تحميل DeepSeek-VL2 ==========
sys.path.insert(0, r"C:\Users\Ahmed\Desktop\KRJAM\DeepSeek-VL2")
try:
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
except Exception:
    from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

MODEL_ID = "deepseek-ai/deepseek-vl2-tiny"
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DTYPE = torch.float16 if USE_CUDA else torch.float32

def _load_model():
    proc = DeepseekVLV2Processor.from_pretrained(MODEL_ID)
    mdl = DeepseekVLV2ForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True
    ).eval().to(DEVICE)
    return proc, mdl

try:
    processor, model = _load_model()
except Exception:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
    processor, model = _load_model()

# ========= ثوابت RAG =========
RAG_DIR = ROOT / "rag"
RAG_INDEX = RAG_DIR / "index.npz"
RAG_BUILD = RAG_DIR / "build_index.py"
RAG_SCORE_MIN = 0.20
RAG_TOP_USED  = 5
RAG_MODE_DEFAULT = os.getenv("RAG_MODE", "auto").lower()  # "always" | "question_only" | "auto" | "adaptive"
DEFAULT_Q = "Describe this image in simple medical terms.".lower()

# ========= السجل =========
HISTORY_FILE = DATA_DIR / "history.json"

def _load_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def _save_history(entries: List[Dict[str, Any]]) -> None:
    HISTORY_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

def _append_history(entry: Dict[str, Any]) -> None:
    hx = _load_history()
    hx.insert(0, entry)
    _save_history(hx)

def _ensure_rag_index() -> None:
    try:
        need_build = not RAG_INDEX.exists()
        if not need_build:
            age_days = (time.time() - RAG_INDEX.stat().st_mtime) / 86400.0
            need_build = age_days > 7
        if need_build and RAG_BUILD.exists():
            subprocess.run([sys.executable, str(RAG_BUILD)], cwd=str(RAG_DIR), check=True)
    except Exception as e:
        print("RAG auto-build failed:", e)

# ========= أدوات =========
def _decode_ids(ids: torch.Tensor, tokenizer, prompt_len):
    if prompt_len is not None and prompt_len < ids.shape[0]:
        ids = ids[prompt_len:]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()

def _fmt(p: Optional[float]) -> str:
    s = f"{p:.4f}".rstrip('0').rstrip('.')
    return s or "0"

def _conf_band(p: Optional[float]) -> str:
    if p is None: return "Unknown"
    if p >= 0.90: return "High"
    if p >= 0.70: return "Medium"
    return "Low"

def _canon_label(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = str(name).lower().strip().replace(" ", "_")
    alias = {
        "notumor": "no_tumor",
        "no_tumor": "no_tumor",
        "no-tumor": "no_tumor",
        "no tumor": "no_tumor",
        "glioma": "glioma",
        "meningioma": "meningioma",
        "pituitary": "pituitary",
    }
    return alias.get(n, n)

def _safe_ext(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext in ALLOWED_EXTS else ext

# ========= [MRI-GATE] =========
def _file_looks_dicom(path: str) -> bool:
    """يتحقق من وجود توقيع DICOM (DICM) عند الإزاحة 128."""
    try:
        with open(path, "rb") as f:
            header = f.read(132)
        return len(header) >= 132 and header[128:132] == b"DICM"
    except Exception:
        return False

def _dicom_to_pil(ds) -> Image.Image:
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + inter
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-6: arr = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr = (arr - mn) / (mx - mn) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if img.mode != "L":
        img = img.convert("L")
    return ImageOps.grayscale(img).convert("RGB")

def _image_colorfulness_score(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    return float(np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))

# عتبة اللون: صور الدماغ MRI عادةً رمادية؛ الصور الملونة غالباً ليست شريحة MRI
MRI_COLORFULNESS_MAX = 18.0
MRI_MIN_SIDE = 32  # أصغر بعد للصورة (تجنب أيقونات أو صور صغيرة جداً)

def _is_mri_image(tmp_path: str) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    يتحقق إن كانت الصورة قابلة لاعتبارها شريحة MRI دماغ:
    - DICOM: يشترط Modality == MR.
    - PNG/JPG/..: يشترط أن تكون ذات ألوان قليلة (شبه رمادية) وحجم معقول.
    """
    # ─── 1) DICOM ───
    if _file_looks_dicom(tmp_path) or (tmp_path and tmp_path.lower().endswith(".dcm")):
        if not _HAS_PYDICOM:
            return (False, "DICOM file detected but pydicom is not installed. Install with: pip install pydicom", None)
        try:
            # قراءة سريعة للموداليتي أولاً دون تحميل البكسلات
            ds = pydicom.dcmread(tmp_path, force=True, stop_before_pixels=True)
            modality = (str(getattr(ds, "Modality", "") or "")).strip().upper()
            if modality != "MR":
                mod_name = modality if modality else "UNKNOWN"
                return (False, f"This DICOM file is not MRI (modality: {mod_name}). Only brain MRI (MR) is supported.", None)
            # تحميل البكسلات فقط للملفات MR
            ds = pydicom.dcmread(tmp_path, force=True, stop_before_pixels=False)
            pil_img = _dicom_to_pil(ds)
            return (True, None, pil_img)
        except Exception as e:
            return (False, f"DICOM read error: {type(e).__name__}: {str(e)[:120]}", None)

    # ─── 2) صور عادية (PNG, JPG, ...) ───
    try:
        pil = Image.open(tmp_path).convert("RGB")
    except UnidentifiedImageError:
        return (False, "Unsupported or corrupt image format. Use PNG, JPEG, TIFF, or DICOM (.dcm).", None)
    except Exception as e:
        return (False, f"Could not open image: {type(e).__name__}", None)

    w, h = pil.size
    if w < MRI_MIN_SIDE or h < MRI_MIN_SIDE:
        return (False, f"Image too small ({w}×{h}). Brain MRI slices are usually at least 64×64 pixels.", None)

    score = _image_colorfulness_score(pil)
    if score > MRI_COLORFULNESS_MAX:
        return (False, f"Image looks like a full-color photo (score {score:.1f}). Brain MRI slices are typically grayscale.", None)

    return (True, None, pil)

# ---------- Firebase Admin (Auth) ----------
import firebase_admin
from firebase_admin import credentials, auth as fb_auth
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH", "firebase_credentials.json")
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        # لا توقف السيرفر في التطوير لو الملف غير موجود
        print("WARN: Firebase credentials not loaded:", e)

def _verify_token_from_request():
    """يرجع dict لمستخدم Firebase أو None"""
    try:
        id_token = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            id_token = auth_header.split("Bearer ", 1)[1]
        session_cookie = request.cookies.get("fb_session")
        if id_token and fb_auth:
            try:
                return fb_auth.verify_id_token(id_token, clock_skew_seconds=60)  # أقصى سماح 60 ثانية
            except TypeError:
                return fb_auth.verify_id_token(id_token)
        if session_cookie and fb_auth:
            return fb_auth.verify_session_cookie(session_cookie, check_revoked=True)
    except Exception as e:
        print("auth verify error:", e)
    return None

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = _verify_token_from_request()
        if user:
            request.user = user
            return f(*args, **kwargs)
        return redirect(url_for("login_page"))
    return wrapper

# ---------- صفحات ----------
@app.get("/login")
def login_page():
    return render_template("login.html", current_page="login")

@app.post("/auth/verify")
def auth_verify():
    data = request.get_json(force=True, silent=True) or {}
    id_token = data.get("idToken")
    if not id_token or not fb_auth:
        return jsonify({"ok": False, "error": "missing idToken or firebase not configured"}), 400

    # مساعد صغير يستعمل skew=60
    def _verify_with_skew():
        try:
            return fb_auth.verify_id_token(id_token, clock_skew_seconds=60)
        except TypeError:
            return fb_auth.verify_id_token(id_token)

    try:
        try:
            decoded = _verify_with_skew()
        except Exception as e1:
            if "Token used too early" in str(e1):
                time.sleep(1.5)  # انتظر قليلاً ثم أعد المحاولة مرة واحدة
                decoded = _verify_with_skew()
            else:
                raise

        # مدة الجلسة
        expires_in = timedelta(days=5)
        session_cookie = fb_auth.create_session_cookie(id_token, expires_in=expires_in)

        # اضبط secure تلقائيًا حسب البروتوكول (False على 127.0.0.1، True على HTTPS)
        proto = request.headers.get("X-Forwarded-Proto", "http")
        is_https = request.is_secure or (proto == "https")
        secure_cookie = is_https

        resp = jsonify({"ok": True, "uid": decoded["uid"], "email": decoded.get("email")})
        resp.set_cookie(
            "fb_session",
            session_cookie,
            max_age=int(expires_in.total_seconds()),
            httponly=True,
            secure=secure_cookie,
            samesite="Lax"
        )
        return resp
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 401

@app.post("/logout")
def logout():
    resp = jsonify({"ok": True})
    resp.delete_cookie("fb_session")
    return resp

@app.get("/")
def home():
    return redirect(url_for("analyze_page"))

@app.get("/analyze")
@require_auth
def analyze_page():
    _ensure_rag_index()
    return render_template("analyze.html", current_page="analyze")

@app.get("/history")
@require_auth
def history_page():
    return render_template("history.html", current_page="history")

@app.get("/help")
def help_page():
    return render_template("help.html", current_page="help")

@app.get("/about")
def about_page():
    return render_template("about.html", current_page="about")

@app.get("/settings")
@require_auth
def settings_page():
    return render_template("settings.html", current_page="settings")

@app.get("/theme")
@require_auth
def theme_page():
    return render_template("theme.html", current_page="theme")

# ---------- Health / Time ----------
@app.get("/healthz")
def healthz():
    return jsonify({"ok": True, "device": str(DEVICE), "cuda": USE_CUDA})

@app.get("/api/time")
def api_time():
    return jsonify({"server_ts": int(time.time())})

# ---------- REST APIs للتاريخ ----------
def _history_add(entry: Dict[str, Any]) -> None:
    try:
        _append_history(entry)
    except Exception as e:
        print("history save error:", e)

@app.get("/api/history")
@require_auth
def api_history_list():
    return jsonify({"ok": True, "items": _load_history()})

@app.post("/api/history/clear")
@require_auth
def api_history_clear():
    _save_history([])
    return jsonify({"ok": True})

# ---------- أدوات إخراج احترافي ----------
def _infer_sequences_from_text(t: str) -> List[str]:
    t = (t or "").lower()
    seq = []
    for k in ("t1", "t2", "flair", "dwi", "adc", "swi"):
        if k in t: seq.append(k.upper())
    return list(dict.fromkeys(seq))  # unique, keep order

def _dedupe_rag_sources(sources: List[Dict[str, Any]], top_n: int = 4) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for s in sources or []:
        k = (s.get("source"), s.get("chunk_id"))
        if k in seen: continue
        seen.add(k)
        out.append(s)
    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out[:top_n]

def _build_patient_text(primary_label: Optional[str], confidence: Optional[float]) -> str:
    """نص موجز مفهوم للعامة."""
    if not primary_label:
        return (
            "In simple terms:\n"
            "- The image was analyzed by AI.\n"
            "- This single image is not enough for a final answer.\n"
            "- Always follow up with your doctor for a full assessment."
        )
    label = primary_label.replace("_", " ").title()
    band = _conf_band(confidence) if confidence is not None else "Unknown"
    if primary_label == "no_tumor":
        return (
            "In simple terms:\n"
            "- The AI did not see signs of a tumour in this slice.\n"
            f"- Result strength: {band}. This is from one image only.\n"
            "- Your doctor will give you the final interpretation."
        )
    return (
        "In simple terms:\n"
        f"- The AI notes features that may point to {label}.\n"
        f"- Result strength: {band}. This is not a final diagnosis.\n"
        "- Please see your doctor for a full report and next steps."
    )

def _build_clinician_text(case_id: str,
                          llm_text: str,
                          cls_name: Optional[str],
                          conf: Optional[float],
                          rag_sources: List[Dict[str, Any]],
                          seq_inferred: List[str],
                          consistency_note: Optional[str]) -> str:
    canon = _canon_label(cls_name) or cls_name
    conf_line = f"{conf:.4f} ({_conf_band(conf)})" if conf is not None else "Unknown"
    sep = "────────────────────────────────────"
    blank = ""

    # ملخص بسيط للمريض في أعلى التقرير
    simple = _build_patient_text(canon, conf)

    parts = [
        sep,
        f"  CASE #{case_id}",
        sep,
        blank,
        "▸ IN SIMPLE TERMS (for you)",
        blank,
        simple,
        blank,
        "▸ FINDINGS & IMPRESSION",
        blank,
        (llm_text.strip() if llm_text else "Findings:\n- Non-diagnostic.\n\nImpression:\n- Uncertain.\n\nNext steps:\n- Radiology review."),
        blank,
    ]

    if seq_inferred:
        parts.extend(["▸ SEQUENCES INFERRED", f"  {', '.join(seq_inferred)}", blank])

    parts.extend([
        "▸ WHAT THE AI DETECTED",
        f"  • Label: {canon or 'Unknown'}",
        f"  • Confidence: {conf_line}",
        blank,
    ])

    caution = _findings_classifier_contradiction(llm_text, canon, conf)
    if caution:
        parts.extend(["▸ CAUTION (possible contradiction)", f"  • {caution}", blank])

    if consistency_note:
        parts.extend(["▸ CONSISTENCY", f"  • {consistency_note}", blank])

    if rag_sources:
        parts.append("▸ REFERENCES (RAG)")
        for s in rag_sources:
            parts.append(f"  • ({s['score']}) {s['source']}#{s['chunk_id']}")
        parts.append(blank)

    parts.extend([
        "▸ IMPORTANT TO KNOW",
        "  • Based on one slice only; sequences are inferred, not confirmed.",
        "  • This is an aid for discussion with your doctor — not a final report.",
    ])
    return "\n".join(parts)

def _consistency_check(canon_label: Optional[str], rag_sources: List[Dict[str, Any]]) -> Optional[str]:
    if not canon_label or not rag_sources:
        return None
    src_names = " ".join([str(s.get("source","")).lower() for s in rag_sources])
    if canon_label == "no_tumor" and any(w in src_names for w in ["glioma","meningioma","pituitary"]):
        return "RAG mentions tumor features while classifier suggests no_tumor."
    if canon_label in {"glioma","meningioma","pituitary"} and any(w in src_names for w in ["no_tumor","no tumor","normal"]):
        return f"RAG mentions non-tumor while classifier suggests {canon_label}."
    return "Aligned (no direct contradiction detected)."

def _findings_classifier_contradiction(llm_text: Optional[str], canon_label: Optional[str], conf: Optional[float]) -> Optional[str]:
    """تنبيه عند تعارض واضح بين نص الـ Findings والصنف من المصنّف (عند ثقة عالية)."""
    if not canon_label or not llm_text or (conf is not None and conf < 0.70):
        return None
    t = llm_text.lower()
    tumor_terms = ["tumor", "tumour", "lesion", "mass", "malignancy", "malignant", "glioma", "meningioma", "pituitary", "abnormality", "patholog"]
    no_tumor_terms = ["no tumor", "no tumour", "no lesion", "normal", "no mass", "no abnormality", "unremarkable"]
    if canon_label == "no_tumor":
        if any(w in t for w in tumor_terms):
            return "The written findings mention possible tumour/lesion, while the classifier suggests no tumour. Please correlate clinically."
    elif canon_label in {"glioma", "meningioma", "pituitary"}:
        if any(w in t for w in no_tumor_terms):
            return f"The written findings suggest normality, while the classifier suggests {canon_label}. Please correlate clinically."
    return None

# ---------- نقطة التحليل ----------
@app.post("/analyze")
@require_auth
def analyze():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "no_file"}), 400

    f = request.files["file"]
    filename = getattr(f, "filename", "uploaded.png") or "uploaded.png"
    ext = _safe_ext(filename)
    if ext and ext not in ALLOWED_EXTS:
        return jsonify({"ok": False, "error": f"unsupported_ext:{ext}"}), 400

    q = request.form.get("question") or "Describe this image in simple medical terms."
    use_rag = request.form.get("use_rag", "true").lower() in {"1","true","yes","on"}
    adv_report = request.form.get("adv_report", "false").lower() in {"1","true","yes","on"}
    allow_non_mri = request.form.get("allow_non_mri", "false").lower() in {"1","true","yes","on"}
    rag_mode_req = (request.form.get("rag_mode") or RAG_MODE_DEFAULT).lower()

    ctype = f.mimetype or ""
    allowed_mimes = {"image/png", "image/jpeg", "image/tiff", "application/dicom", "application/dicom+json", "application/octet-stream"}
    if ctype and ctype not in allowed_mimes:
        print("warn: suspicious content-type", ctype)

    suf = ext if ext else ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        # [MRI-GATE]
        is_mri, reason, preloaded_img = _is_mri_image(tmp_path)
        if not is_mri and not allow_non_mri:
            msg = f"This image was not accepted as a brain MRI.\n\n{reason or 'Unknown reason.'}"
            return jsonify({
                "ok": True,
                "text": msg,
                "detail": reason or "unknown_reason",
                "rag": {"used": False, "sources": []},
                "heatmap": None,
                "adv": {"enabled": False, "gradcam_error": None}
            }), 200

        image = preloaded_img if preloaded_img is not None else Image.open(tmp_path).convert("RGB")

        # ===== 1) Classifier (اختياري)
        cls_name, conf, class_scores = None, None, None
        try:
            from classify_infer import infer_image
            cls_name, conf, class_scores = infer_image(image)
        except Exception:
            pass
        cls_name_canon = _canon_label(cls_name)

        # ===== 2) RAG
        def _should_run_rag(mode: str) -> bool:
            if not use_rag:
                return False
            has_real_question = q and q.strip() and q.strip().lower() != DEFAULT_Q
            cls_conf_ok = (cls_name_canon is not None) and (conf is not None) and (conf >= 0.60)
            if mode == "always":
                return True
            if mode == "question_only":
                return bool(has_real_question)
            # "adaptive" = نفس "auto": RAG عند سؤال حقيقي أو ثقة المصنّف ≥ 0.60
            if mode == "adaptive":
                return bool(has_real_question or cls_conf_ok)
            return bool(has_real_question or cls_conf_ok)

        rag_payload: Dict[str, Any] = {"used": False, "sources": []}
        context_text = ""
        if _should_run_rag(rag_mode_req):
            _ensure_rag_index()
            try:
                from rag.retriever import RAGRetriever
                RAG = RAGRetriever()

                extra_kw = ["MRI", "T1", "T2", "FLAIR", "lesion", "enhancement", "edema", "tumor"]
                bias_txt = f"Classifier suggests {cls_name_canon} (p={conf:.2f}). " if (cls_name_canon and conf is not None) else ""
                composed_query = (bias_txt + q + " " + " ".join(extra_kw)).strip()

                raw_hits: List[Dict[str, Any]] = RAG.search(composed_query, top_k=12)
                def _sim(h): return float(h.get("sim", h.get("score", 0.0)))
                hits = list(raw_hits)

                def _contains_any(text: str, terms: List[str]) -> bool:
                    t = (text or "").lower()
                    return any(w in t for w in terms)

                if cls_name_canon and conf is not None and conf >= 0.90:
                    if cls_name_canon == "no_tumor":
                        penalize_terms = ["glioma", "meningioma", "pituitary", "metastasis", "tumor"]
                    else:
                        penalize_terms = ["no_tumor", "no tumor", "normal"]

                    for h in hits:
                        src_text = (h.get("source") or "") + " " + (h.get("text") or "")
                        penalty = 0.0
                        if _contains_any(src_text, penalize_terms):
                            penalty -= 0.20
                        if cls_name_canon != "no_tumor" and _contains_any(src_text, [cls_name_canon]):
                            penalty += 0.08
                        h["score"] = float(h.get("score", h.get("sim", 0.0))) + penalty
                    hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)

                # فلترة وتخفيف
                hits = [h for h in hits if _sim(h) >= RAG_SCORE_MIN]
                if not hits and raw_hits:
                    auto_min = max(0.0, RAG_SCORE_MIN - 0.15)
                    hits = [h for h in raw_hits if _sim(h) >= auto_min]
                if not hits and raw_hits:
                    hits = raw_hits[:1]

                # تبسيط المسارات + تقريب السكور
                for h in hits:
                    short_src = os.path.basename(h.get("source", "")) or h.get("source", "")
                    h["source"] = short_src
                    h["score"] = round(_sim(h), 3)

                rag_payload["used"] = len(hits) > 0
                rag_payload["sources"] = [
                    {"source": h["source"], "chunk_id": h["chunk_id"], "score": h["score"]}
                    for h in hits[:RAG_TOP_USED]
                ]

                # نص سياقي إلى الـ LLM
                top = hits[:RAG_TOP_USED]
                bullets = "\n".join([f"- {h.get('text','')}" for h in top if h.get('text')])
                if bullets:
                    context_text = (
                        "Context (retrieved passages):\n"
                        f"{bullets}\n\n"
                        "Task: Write concise Findings / Impression / Next steps. "
                        "Use ONLY the context above; if unsure, say you are unsure."
                    )
            except Exception as e:
                print("RAG error:", e)

        # إزالة تكرار RAG
        rag_payload["sources"] = _dedupe_rag_sources(rag_payload["sources"], top_n=4)

        # ===== 3) DeepSeek-VL2
        sys_prompt = (context_text + "\n\n") if context_text else ""
        messages = [
            {"role": "<|User|>", "content": sys_prompt + "<image>\n" + q, "images": [tmp_path]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        inputs = processor(conversations=messages, images=[image], force_batchify=True).to(DEVICE)
        if hasattr(inputs, "images") and isinstance(inputs.images, torch.Tensor): inputs.images = inputs.images.to(DTYPE)
        if hasattr(inputs, "pixel_values") and isinstance(inputs.pixel_values, torch.Tensor): inputs.pixel_values = inputs.pixel_values.to(DTYPE)
        if hasattr(inputs, "input_ids") and isinstance(inputs.input_ids, torch.Tensor): inputs.input_ids = inputs.input_ids.to(torch.int64)

        bos_id = getattr(processor.tokenizer, "bos_token_id", None)
        pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        eos_id = processor.tokenizer.eos_token_id
        if bos_id is None: bos_id = eos_id

        with torch.inference_mode():
            inputs_embeds = model.prepare_inputs_embeds(**inputs) if hasattr(model, "prepare_inputs_embeds") else None
            gen_kwargs = dict(
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eps_id if (eps_id:=eos_id) else eos_id,
                bos_token_id=bos_id,
            )
            if inputs_embeds is not None:
                out = model.generate(inputs_embeds=inputs_embeds, **gen_kwargs)
            else:
                out = model.generate(input_ids=inputs.input_ids, **gen_kwargs)

        gen_ids = out[0]
        prompt_len = inputs.input_ids.shape[1] if hasattr(inputs, "input_ids") else None
        llm_text_raw = _decode_ids(gen_ids, processor.tokenizer, prompt_len).strip()

        # ===== 4) فرض بنية Findings/Impression/Next steps لو ناقصة
        def _force_structured(text: str) -> str:
            t = (text or "").strip()
            tl = t.lower()
            has_find = "findings" in tl
            has_impr = "impression" in tl
            has_next = "next steps" in tl or "recommendations" in tl
            if has_find and has_impr and has_next:
                return t
            blocks = []
            blocks.append("Findings:\n- " + (t if t else "Uncertain; limited image context."))
            if cls_name and conf is not None:
                blocks.append(f"Impression:\n- Model suggests **{_canon_label(cls_name) or cls_name}** (p≈{_fmt(conf)}). Correlate clinically.")
            else:
                blocks.append("Impression:\n- Non-diagnostic description. Consider radiologist review.")
            blocks.append("Next steps:\n- Review with neuroradiology.\n- Consider multi-sequence MRI & clinical correlation.")
            return "\n\n".join(blocks)

        llm_text = _force_structured(llm_text_raw)

        # ===== 5) Grad-CAM (اختياري)
        heatmap_b64, gradcam_error = None, None
        if adv_report:
            try:
                from classify_infer import infer_image as _infer_cls, IDX_TO_NAME, CKPT_PATH
                if cls_name is None:
                    try:
                        cls_name, conf, class_scores = _infer_cls(image)
                        cls_name_canon = _canon_label(cls_name)
                    except Exception:
                        pass
                from gradcam import make_gradcam_on_pil
                heatmap_b64 = make_gradcam_on_pil(
                    pil_img=image,
                    ckpt_path=str(CKPT_PATH),
                    num_classes=len(IDX_TO_NAME),
                    device=("cuda" if torch.cuda.is_available() else "cpu")
                )
            except Exception as e:
                gradcam_error = f"{type(e).__name__}: {e}"

        # ===== 6) إخراج احترافي
        case_id = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        seq_inferred = _infer_sequences_from_text(llm_text) or _infer_sequences_from_text(q)
        rag_sources_final = rag_payload["sources"]
        consistency_note = _consistency_check(_canon_label(cls_name), rag_sources_final)

        clinician_text = _build_clinician_text(
            case_id=case_id,
            llm_text=llm_text,
            cls_name=cls_name,
            conf=conf,
            rag_sources=rag_sources_final,
            seq_inferred=seq_inferred,
            consistency_note=consistency_note
        )
        patient_text = _build_patient_text(_canon_label(cls_name), conf)

        # JSON مُهيكل
        report_json = {
            "case_id": case_id,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "filename": filename,
            "series_inferred": seq_inferred,
            "findings_text": llm_text,
            "impression": {
                "primary": _canon_label(cls_name),
                "confidence": float(conf) if conf is not None else None,
                "band": _conf_band(conf) if conf is not None else "Unknown",
            },
            "differential": [],
            "recommendations": [],
            "classifier": {
                "label": _canon_label(cls_name) or cls_name,
                "confidence": float(conf) if conf is not None else None
            },
            "rag": {
                "used": bool(rag_sources_final),
                "sources": rag_sources_final
            },
            "consistency": {
                "note": consistency_note or "Unknown"
            },
            "limitations": [
                "Single-slice view",
                "Auto-inferred sequences",
                "No contrast confirmation"
            ],
            "disclaimer": "Automated assistive summary; not a final radiology report."
        }

        # حفظ ملخّص بسيط في التاريخ
        preview = (llm_text.splitlines()[0] if llm_text else "")[:140]
        _history_add({
            "ts": report_json["timestamp"],
            "case_id": case_id,
            "filename": filename,
            "label": _canon_label(cls_name) or cls_name,
            "confidence": float(conf) if conf is not None else None,
            "text_preview": preview,
            "rag_used": bool(rag_payload["used"]),
            "rag_sources": rag_sources_final,
            "adv_enabled": bool(adv_report)
        })

        return jsonify({
            "ok": True,
            "text": clinician_text,
            "text_patient": patient_text,
            "report_json": report_json,
            "rag": {"used": bool(rag_sources_final), "sources": rag_sources_final},
            "heatmap": heatmap_b64,
            "adv": {"enabled": adv_report, "gradcam_error": gradcam_error}
        })

    except UnidentifiedImageError:
        return jsonify({"ok": False, "error": "invalid_image"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ---------- أخطاء ----------
@app.errorhandler(413)
def too_large(e):
    return jsonify({"ok": False, "error": f"file_too_large_max_{MAX_CONTENT_MB}_MB"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"ok": False, "error": "not_found"}), 404

# ---------- تشغيل ----------
if __name__ == "__main__":
    # التزم بفتح http://127.0.0.1:5000/login أثناء التطوير
    app.run(host="127.0.0.1", port=5000, debug=False)
