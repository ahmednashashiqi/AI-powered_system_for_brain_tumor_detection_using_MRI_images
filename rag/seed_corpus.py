# -*- coding: utf-8 -*-
from pathlib import Path

ROOT = Path(__file__).resolve().parent
pairs = {
"corpus/tumor_types/glioma.txt": """Gliomas are primary brain tumors from glial cells. On MRI they are often hyperintense on T2/FLAIR with irregular, infiltrative margins and frequent peritumoral edema. High-grade gliomas may show heterogeneous enhancement and necrosis.""",
"corpus/tumor_types/meningioma.txt": """Meningiomas arise from arachnoid cap cells and are usually extra-axial, dural-based masses. On MRI they often show isointense T1, variable T2, strong homogeneous enhancement, and may demonstrate a “dural tail” sign.""",
"corpus/tumor_types/pituitary.txt": """Pituitary adenomas originate in the anterior pituitary. Microadenomas are <10 mm; macroadenomas ≥10 mm. On MRI they are typically T1 isointense with variable enhancement and can cause mass effect on the optic chiasm.""",
"corpus/tumor_types/no_tumor.txt": """No-tumor cases show normal brain parenchyma without focal mass effect or abnormal enhancement. Ventricles and sulci are age-appropriate; no restricted diffusion or pathologic T2/FLAIR hyperintensity.""",
"corpus/anatomy/t1_mri.txt": """T1-weighted MRI provides high anatomic detail. Fat is bright, CSF is dark, white matter is brighter than gray matter. Post-contrast T1 highlights areas with BBB disruption and solid tumor enhancement.""",
"corpus/anatomy/t2_mri.txt": """T2-weighted MRI is fluid-sensitive: CSF and edema appear bright. Many tumors and inflammatory lesions are hyperintense, improving conspicuity of perilesional changes.""",
"corpus/anatomy/flair_mri.txt": """FLAIR suppresses free CSF to reveal periventricular and cortical/subcortical lesions. It improves detection of edema adjacent to tumors that may be obscured on conventional T2.""",
"corpus/anatomy/diffusion_adc.txt": """DWI detects restricted diffusion; ADC maps confirm it. Acute infarcts show high DWI/low ADC. Some tumors or abscesses may restrict. SWI is sensitive to blood products and calcifications.""",
"corpus/explainability/gradcam.txt": """Grad-CAM highlights regions that most influenced a model’s decision by weighting feature-map gradients. In neuro-MRI it localizes tumor core or edema, aiding interpretability.""",
"corpus/explainability/xai_in_medicine.txt": """Explainable AI improves trust by exposing model reasoning. Visual explanations, confidence bands, and references help clinicians judge AI outputs.""",
"corpus/explainability/deepseek_overview.txt": """DeepSeek-VL2 is a vision-language model for image-grounded answers. With RAG, textual outputs can be grounded in curated neuro-MRI knowledge.""",
"corpus/tips/reporting.txt": """Radiology reports follow: Findings, Impression, and Recommendations. Findings are objective; Impression synthesizes diagnosis; Recommendations guide next steps."""
}
for rel, content in pairs.items():
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip() + "\n", encoding="utf-8")
print(f"Seeded {len(pairs)} files under {ROOT/'corpus'}")
