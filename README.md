# AI-Powered Brain MRI Tumor Detection System

An AI-powered web application for automatic brain tumor detection, classification, and explanation using MRI images.

The system combines a custom EfficientNet-B0 classifier, Grad-CAM explainability, and a Retrieval-Augmented Generation (RAG) pipeline powered by DeepSeek-VL2-27B to provide both prediction results and medical reasoning.

---

# Features

* Brain MRI tumor classification
* Supports 4 classes:

  * Glioma
  * Meningioma
  * Pituitary Tumor
  * No Tumor
* Explainable AI using Grad-CAM heatmaps
* Natural-language medical explanation using DeepSeek-VL2-27B + RAG
* Secure user authentication with Firebase
* Upload MRI images through a simple web interface
* Real-time prediction and report generation
* History page to store previous analyses

---

# System Architecture

The project consists of three main layers:

1. Front-End

   * HTML
   * CSS
   * JavaScript

2. Back-End

   * Flask (Python)
   * REST API
   * Firebase Authentication and Storage

3. AI Engine

   * EfficientNet-B0 for tumor classification
   * Grad-CAM for visual explanation
   * DeepSeek-VL2-27B for text reasoning
   * RAG module with local medical knowledge base

---

# Dataset

The model was trained on public MRI datasets collected from Kaggle.

Main classes:

| Class      | Description                                       |
| ---------- | ------------------------------------------------- |
| Glioma     | Irregular brain tumor with invasive appearance    |
| Meningioma | Well-defined tumor attached to the brain covering |
| Pituitary  | Tumor located near the pituitary gland            |
| No Tumor   | Healthy brain MRI                                 |

Dataset split:

* 80% Training
* 20% Testing

Image preprocessing includes:

* Resize to 224 × 224
* Contrast enhancement
* Random flip augmentation
* Random rotation augmentation

---

# Technologies Used

* Python 3.10+
* Flask
* PyTorch
* Torchvision
* OpenCV
* Grad-CAM
* Firebase
* DeepSeek-VL2-27B
* HTML / CSS / JavaScript
* CUDA 11.8

---

# Model Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Model         | EfficientNet-B0  |
| Epochs        | 15               |
| Batch Size    | 32               |
| Learning Rate | 0.0001           |
| Optimizer     | AdamW            |
| Loss Function | CrossEntropyLoss |
| Input Size    | 224 × 224        |

---

# Performance

Internal testing achieved:

* Accuracy: 99.2%
* Strong Precision, Recall, and F1-score across all classes

External testing on unseen MRI datasets achieved:

* Accuracy: 65.1%

This shows the effect of domain shift and highlights the importance of testing medical AI systems on external datasets.

---

# Folder Structure

```text
project/
│
├── app.py
├── requirements.txt
├── static/
│   ├── css/
│   ├── js/
│   ├── uploads/
│   └── heatmaps/
│
├── templates/
│   ├── login.html
│   ├── analyze.html
│   ├── history.html
│   └── settings.html
│
├── model/
│   ├── efficientnet_model.pth
│   ├── classifier.py
│   └── gradcam.py
│
├── rag/
│   ├── glioma.txt
│   ├── meningioma.txt
│   ├── pituitary.txt
│   ├── notumor.txt
│   └── tumor_grading.txt
│
├── dataset/
├── firebase/
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

## 2. Create virtual environment

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Add Firebase credentials

Place your Firebase service account JSON file inside the project folder:

```text
firebase/firebase_credentials.json
```

## 5. Run the application

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

# How It Works

1. The user logs into the web application.
2. The user uploads an MRI image.
3. The system validates the file format.
4. EfficientNet-B0 predicts the tumor class.
5. Grad-CAM generates a heatmap showing important image regions.
6. The RAG module retrieves medical knowledge.
7. DeepSeek-VL2-27B generates an explanation.
8. The final result is displayed with:

   * Tumor type
   * Confidence score
   * Heatmap
   * AI-generated explanation

---

# Example Output

```text
Prediction: Glioma
Confidence: 97.4%

Explanation:
The MRI image shows an irregular lesion with surrounding edema,
which is commonly associated with glioma. The highlighted area in
the Grad-CAM heatmap indicates the region that most influenced the
prediction.
```

---

# Limitations

* The system is designed for research and educational use only.
* It is not intended to replace professional medical diagnosis.
* External dataset performance is lower due to differences in MRI quality and scanner settings.
* Tumor grading is not supported; only tumor type classification is included.

---

# Future Improvements

* Support DICOM metadata analysis
* Add more MRI modalities such as T1, T2, and FLAIR
* Improve external dataset generalization
* Add multilingual explanations
* Integrate more advanced medical knowledge sources
* Support cloud deployment and mobile access

---

# Authors

* Ahmed M. Nashashiqi
* Mohammed A. Algain
* Abdulaziz M. Mulazim

Supervisor:

* Dr. Abdullah Al‑Shanqeeti

---

# License

This project is for academic and research purposes only.

<img width="1906" height="913" alt="image" src="https://github.com/user-attachments/assets/1630a7dd-3d23-4e55-8fda-662a18cc2d63" />

<img width="938" height="766" alt="image" src="https://github.com/user-attachments/assets/3e812890-7f36-4152-8324-b63a92d55442" />

<img width="467" height="865" alt="image" src="https://github.com/user-attachments/assets/1367a008-467b-4c7c-841f-27a9582282e7" />

<img width="1898" height="912" alt="image" src="https://github.com/user-attachments/assets/703eb055-f263-49ea-91ed-99a33d1114e8" />

<img width="1913" height="910" alt="image" src="https://github.com/user-attachments/assets/57a4ba74-a6c6-44da-9d01-35ab7c1e1350" />

<img width="1914" height="911" alt="image" src="https://github.com/user-attachments/assets/b49b6bc7-7cf8-43cb-8ba0-9ef0f16f79a9" />
