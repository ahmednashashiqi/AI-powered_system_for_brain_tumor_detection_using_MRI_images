# -*- coding: utf-8 -*-
r"""
analyze_mri.py
تشغيل DeepSeek-VL2 على صورة (MRI أو غيرها) مع معالجة صارمة للـ dtypes لمنع
RuntimeError: Input type (BFloat16) and bias type (Half) should be the same

تشغيل (PowerShell):
  cd C:\Users\Ahmed\Desktop\KRJAM
  .\.dsenv\Scripts\activate
  python analyze_mri.py --image ".\images\MidlineGlioma.jpg" --question "Analyze this brain MRI briefly."

ملاحظات:
- الكود يفرض FP16 على GPU ويغلق الـ autocast حتى لا يتحول شيء إلى BF16.
- على CPU يستعمل FP32 تلقائيًا.
- إن احتجت التبديل إلى صورة أخرى: استخدم --image "path\to\image.jpg"
"""

import os
import sys
import argparse
from typing import Optional

# اجبر بايثون يستخدم نسخة الريبو المحلي (لو مثبتة ككود)
sys.path.insert(0, r"C:\Users\Ahmed\Desktop\KRJAM\DeepSeek-VL2")

print("🔹 Importing DeepSeek-VL2...")
try:
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
except Exception:
    from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
print("✅ DeepSeek-VL2 imported successfully!")

import torch
from PIL import Image


def decode_ids(ids: torch.Tensor, tokenizer, prompt_len: Optional[int]) -> str:
    """تحويل المخرجات إلى نص"""
    if prompt_len is not None and prompt_len < ids.shape[0]:
        ids = ids[prompt_len:]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-VL2 MRI/Image analyzer with strict FP16 on CUDA")
    parser.add_argument("--image", type=str, required=False,
                        default=r"C:\Users\Ahmed\Desktop\KRJAM\images\MidlineGlioma.jpg",
                        help="Path to input image")
    parser.add_argument("--question", type=str, required=False,
                        default="Describe this image in simple medical terms.",
                        help="Question to ask the model")
    parser.add_argument("--model-id", type=str, required=False,
                        default="deepseek-ai/deepseek-vl2-tiny",
                        help="HF model id (e.g., deepseek-ai/deepseek-vl2 or -tiny)")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    # اختيار الجهاز والنوع
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if use_cuda else torch.float32
    print(f"🔹 Device: {device} | dtype: {dtype}")

    if use_cuda:
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True

    # تحميل النموذج والمعالج
    processor = DeepseekVLV2Processor.from_pretrained(args.model_id)
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).eval().to(device)
    print("✅ Model and processor loaded successfully!")

    # التحقق من الصورة
    img_path = args.image
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    image = Image.open(img_path).convert("RGB")

    # نبني محادثة
    messages = [
        {"role": "<|User|>", "content": "<image>\n" + args.question, "images": [img_path]},
        {"role": "<|Assistant|>", "content": ""},
    ]

    # تجهيز المدخلات
    inputs = processor(conversations=messages, images=[image], force_batchify=True).to(device)

    # توحيد الأنواع
    if hasattr(inputs, "images") and isinstance(inputs.images, torch.Tensor):
        inputs.images = inputs.images.to(dtype)
    if hasattr(inputs, "pixel_values") and isinstance(inputs.pixel_values, torch.Tensor):
        inputs.pixel_values = inputs.pixel_values.to(dtype)
    if hasattr(inputs, "input_ids") and isinstance(inputs.input_ids, torch.Tensor):
        inputs.input_ids = inputs.input_ids.to(torch.int64)

    print("🔹 Generating response...")
    with torch.inference_mode():
        bos_id = getattr(processor.tokenizer, "bos_token_id", None)
        pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        eos_id = processor.tokenizer.eos_token_id
        if bos_id is None:
            bos_id = eos_id

        if use_cuda:
            import torch.cuda.amp as amp
            with amp.autocast(enabled=False):  # منع التحويل إلى BF16
                inputs_embeds = model.prepare_inputs_embeds(**inputs) if hasattr(model, "prepare_inputs_embeds") else None
                if inputs_embeds is not None:
                    out = model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=args.max_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                        bos_token_id=bos_id,
                    )
                else:
                    out = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=args.max_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                        bos_token_id=bos_id,
                    )
        else:
            inputs_embeds = model.prepare_inputs_embeds(**inputs) if hasattr(model, "prepare_inputs_embeds") else None
            if inputs_embeds is not None:
                out = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    bos_token_id=bos_id,
                )
            else:
                out = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    bos_token_id=bos_id,
                )

    gen_ids = out[0]
    prompt_len = inputs.input_ids.shape[1] if hasattr(inputs, "input_ids") else None
    text = decode_ids(gen_ids, processor.tokenizer, prompt_len)

    print("\n🧠 Model Output:")
    print(text)
    print("\nNote: This model is general-purpose and not a diagnostic or medical tool.")


if __name__ == "__main__":
    main()
