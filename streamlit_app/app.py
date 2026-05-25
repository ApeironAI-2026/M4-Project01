"""
Pneumonia Detection Streamlit App
Apeiron AI
"""

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
page_title="Pneumonia Detection",
page_icon="🩺"
)

st.title("🩺 Pneumonia Detection")
st.markdown(
"Upload chest X-ray image"
)

class_names=["Normal","Pneumonia"]

model=torch.load(
"../model/best_model.pth",
map_location='cpu'
)

transform=transforms.Compose([
transforms.Resize((224,224)),
transforms.ToTensor()
])

uploaded=st.file_uploader(
"Upload X-ray",
type=['png','jpg','jpeg']
)

if uploaded:

    image=Image.open(uploaded)

    st.image(image)

    x=transform(image)

    x=x.unsqueeze(0)

    with torch.no_grad():

        output=model(x)

        prob=torch.softmax(output,1)

        pred=torch.argmax(prob)

    st.success(
        f"Prediction:{class_names[pred]}"
    )

    st.metric(
    "Confidence",
    f"{prob.max()*100:.2f}%"
    )

    st.markdown("### GradCAM Visualization")