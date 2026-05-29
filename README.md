# 🩺 Pneumonia Detection using CNN vs ResNet50 vs EfficientNetB0

Deep Learning Healthcare AI Project for Chest X-ray Classification

**Apeiron AI — Boundless Possibilities, Infinite Potential**

---

# 📌 Project Overview

This project builds an end-to-end deep learning healthcare system capable of detecting pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs) and transfer learning architectures.

The project compares:

* Custom CNN
* ResNet50
* EfficientNetB0

to identify the most effective model for medical image classification.

The final solution includes:

✅ Data preprocessing
✅ Data augmentation
✅ Deep learning model comparison
✅ Medical evaluation metrics
✅ Grad-CAM explainability
✅ Model serialization
✅ Streamlit deployment for real-time inference

---

# 🏥 Business Problem

Pneumonia is a serious respiratory disease that requires rapid diagnosis and treatment. Manual examination of chest X-rays can be time-consuming and depends heavily on expert radiologists.

AI-powered medical imaging systems can assist healthcare professionals by:

* Accelerating diagnosis
* Improving screening efficiency
* Reducing radiologist workload
* Supporting clinical decision-making
* Detecting pneumonia in resource-limited environments

This project demonstrates how deep learning can support automated pneumonia detection using chest radiographs.

---

# 🎯 Learning Objectives

By completing this project, you will learn how to:

* Build CNN architectures for medical imaging
* Apply transfer learning using pretrained models
* Preprocess chest X-ray datasets
* Perform image augmentation
* Train and evaluate deep learning models
* Compare CNN architectures fairly
* Generate Grad-CAM explainability visualizations
* Save trained models for deployment
* Build healthcare AI interfaces using Streamlit

---

# 📂 Dataset Information

### Dataset Used

Chest X-Ray Images (Pneumonia)

Dataset Source:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

# 📊 Dataset Structure

The dataset contains chest X-ray images divided into two classes:

| Class     | Description                             |
| --------- | --------------------------------------- |
| NORMAL    | Healthy chest X-ray                     |
| PNEUMONIA | Chest X-ray showing pneumonia infection |

Typical structure:

```text id="myg1ee"
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

# 🏗️ Project Structure

```text id="04t1bx"
M4-Project01/
│
├── data/
│   └── chest_xray/
│
├── notebooks/
│   └── Pneumonia_Detection.ipynb
│
├── model/
│   ├── best_model.pth
│   └── config.json
│
├── streamlit_app/
│   ├── app.py
│   └── requirements.txt
│
└── README.md
```

---

# 🧠 Deep Learning Pipeline

## STEP 1 — Data Loading

The chest X-ray dataset is loaded using:

* PyTorch ImageFolder
* DataLoader pipelines

Dataset exploration includes:

* Sample X-ray visualization
* Class distribution analysis
* Image dimension inspection

---

## STEP 2 — Image Preprocessing

Images are preprocessed using:

* Resize to 224×224
* Tensor conversion
* Pixel normalization

Example:

```python id="sv4jxm"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

---

## STEP 3 — Data Augmentation

To improve model generalization and reduce overfitting, augmentation techniques include:

* Horizontal flipping
* Rotation
* Random affine transforms
* Random resized crops

Example:

```python id="0tjlwm"
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05)
    ),
    transforms.ToTensor()
])
```

---

# 🤖 Models Compared

## 1. Custom CNN

A manually designed convolutional neural network built using:

* Convolutional layers
* Max pooling
* ReLU activation
* Fully connected layers

---

## 2. ResNet50

Transfer learning architecture pretrained on ImageNet.

Benefits:

* Deep residual connections
* Strong feature extraction
* Improved convergence

---

## 3. EfficientNetB0

Efficient scaling-based CNN architecture.

Benefits:

* High accuracy
* Lightweight architecture
* Strong medical imaging performance

---

# ⚙️ Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Optimizer     | Adam             |
| Learning Rate | 0.001            |
| Batch Size    | 32               |
| Epochs        | 20–50            |
| Loss Function | CrossEntropyLoss |

---

# 📈 Evaluation Metrics

The models are evaluated using healthcare-focused metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

⚠️ Recall is especially important in medical diagnosis because false negatives can be dangerous.

---

# 📊 Visualizations

The notebook includes:

* Training loss curves
* Validation accuracy curves
* Confusion matrices
* Model comparison charts
* Grad-CAM heatmaps
* Prediction examples

---

# 🔥 Grad-CAM Explainability

Grad-CAM visualizations highlight lung regions influencing model predictions.

This improves:

* Transparency
* Clinical interpretability
* Trust in AI-assisted diagnosis

Example outputs include:

✅ Heatmap overlays
✅ Pneumonia activation regions
✅ Explainable predictions

---

# 🧪 Example Prediction

### Input

Chest X-ray image uploaded through Streamlit interface.

### Output

```text id="mwjlwm"
Prediction:
PNEUMONIA

Confidence:
98.4%
```

---

# 🏆 Results

| Model          | Accuracy | Recall | F1-score | ROC-AUC |
| -------------- | -------- | ------ | -------- | ------- |
| Custom CNN     | TBD      | TBD    | TBD      | TBD     |
| ResNet50       | TBD      | TBD    | TBD      | TBD     |
| EfficientNetB0 | TBD      | TBD    | TBD      | TBD     |

---

# 🧠 Key Insights

* Transfer learning models significantly outperform small CNNs on medical imaging tasks.
* EfficientNetB0 often achieves the best balance between accuracy and efficiency.
* Data augmentation improves generalization on limited medical datasets.
* Explainability is critical in healthcare AI systems.

---

# 💾 Model Saving

The best-performing model is exported for deployment.

Saved files:

```text id="r0i9w0"
best_model.pth
config.json
```

---

# 🚀 Streamlit Deployment

The project includes a Streamlit web application for real-time pneumonia detection.

### Features

✅ Chest X-ray upload
✅ Real-time prediction
✅ Confidence score display
✅ Grad-CAM visualization
✅ Dynamic medical feedback
✅ Fast inference

---

# ▶️ Running the Streamlit App

Navigate to the Streamlit application folder:

```bash id="s5on6k"
cd streamlit_app
```

Run the application:

```bash id="y9ujpm"
streamlit run app.py
```

Expected Output:

```text id="8avmrb"
Local URL: http://localhost:8501
```

---

# 📦 Installation

Clone repository:

```bash id="s0j7rr"
git clone <repository-url>
cd M4-Project01
```

Install dependencies:

```bash id="w89hsp"
pip install -r requirements.txt
```

---

# 📋 requirements.txt

```text id="7d65a9"
torch
torchvision
timm
numpy
pandas
matplotlib
seaborn
scikit-learn
streamlit
pillow
opencv-python
```

---

# 🧪 Reproducibility

Random seeds are configured to ensure reproducible experiments:

```python id="vg8z0z"
torch.manual_seed(42)
np.random.seed(42)
```

---

# ⚠️ Important Notes

* Medical AI systems should assist clinicians, not replace them.
* Consistent preprocessing between training and deployment is critical.
* Grad-CAM improves explainability but does not guarantee clinical correctness.

---

# 🔮 Future Improvements

Potential future enhancements:

* DenseNet121 implementation
* Vision Transformers (ViT)
* Ensemble learning
* Multi-disease chest X-ray classification
* Segmentation-based explainability
* Cloud deployment
* Real-time hospital integration

---

# 🧰 Technologies Used

* Python
* PyTorch
* Torchvision
* TIMM
* NumPy
* Pandas
* Matplotlib
* Seaborn
* OpenCV
* Streamlit

---

# 💼 CV / Resume Description

Developed an end-to-end healthcare AI system for pneumonia detection from chest X-ray images using deep learning and transfer learning architectures (CNN, ResNet50, and EfficientNetB0), including preprocessing, augmentation, model comparison, Grad-CAM explainability, evaluation, and Streamlit deployment for real-time medical image inference.

---

# 👨‍💻 Author

Apeiron AI

---

# 📜 License

This project is for educational and research purposes only.

---

© 2026 Apeiron AI
Boundless Possibilities, Infinite Potential