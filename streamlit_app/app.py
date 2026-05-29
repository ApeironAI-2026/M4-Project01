"""
Pneumonia Detection Streamlit App
Apeiron AI
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🩺 Pneumonia Detection System")
st.markdown(
    """
Upload a chest X-ray image to predict whether the patient has:

- NORMAL lungs
- PNEUMONIA

This AI system uses deep learning with transfer learning.
"""
)

# ---------------------------------------------------
# DEVICE
# ---------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------
# LOAD CONFIG
# ---------------------------------------------------

CONFIG_PATH = "../model/config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

model_name = config["model_name"]
input_size = config["input_size"]
num_classes = config["num_classes"]

# ---------------------------------------------------
# CLASS NAMES
# ---------------------------------------------------

class_names = ["NORMAL", "PNEUMONIA"]

# ---------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------

@st.cache_resource
def load_model():

    if model_name == "ResNet50":

        model = torchvision.models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "EfficientNetB0":

        model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=num_classes
        )

    else:
        raise ValueError("Unsupported model")

    model.load_state_dict(
        torch.load(
            "../model/best_model.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model

model = load_model()

# ---------------------------------------------------
# IMAGE TRANSFORM
# ---------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
])

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["png", "jpg", "jpeg"]
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if uploaded_file is not None:

    try:

        # Load image
        image = Image.open(uploaded_file).convert("RGB")

        # Display image
        st.image(
            image,
            caption="Uploaded Chest X-ray",
            use_container_width=True
        )

        # Transform image
        x = transform(image)
        x = x.unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():

            output = model(x)

            probabilities = torch.softmax(output, dim=1)

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        predicted_class = class_names[prediction.item()]
        confidence_score = confidence.item() * 100

        # Display Results
        st.subheader("Prediction Result")

        if predicted_class == "PNEUMONIA":
            st.error(f"Prediction: {predicted_class}")
        else:
            st.success(f"Prediction: {predicted_class}")

        st.metric(
            "Confidence Score",
            f"{confidence_score:.2f}%"
        )

        # ---------------------------------------------------
        # PROBABILITY SCORES
        # ---------------------------------------------------

        st.subheader("Class Probabilities")

        for i, class_name in enumerate(class_names):

            st.write(
                f"{class_name}: "
                f"{probabilities[0][i].item() * 100:.2f}%"
            )

        # ---------------------------------------------------
        # GRAD-CAM PLACEHOLDER
        # ---------------------------------------------------

        st.subheader("Grad-CAM Visualization")

        st.info(
            "Grad-CAM visualization will appear here."
        )

        # TODO:
        # Add Grad-CAM implementation

    except Exception as e:

        st.error(f"Error processing image: {e}")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")
st.caption(
    "Apeiron AI — Boundless Possibilities, Infinite Potential"
)