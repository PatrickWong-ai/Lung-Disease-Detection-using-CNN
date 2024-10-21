import os
import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import gdown

# Set device to CPU for Streamlit Cloud deployment
device = torch.device("cpu")

# Model weights path and URL
MODEL_PATH = 'resnet50_lung_model.pth'
weights_url = "https://drive.google.com/uc?id=1QWb3w9u2eYPqWFPjSiXSdysnEjvyVG-n"

# Download weights if not found
if not os.path.exists(MODEL_PATH):
    try:
        gdown.download(weights_url, MODEL_PATH, quiet=False)
        st.success("Model weights downloaded successfully.")
    except Exception as e:
        st.error(f"Failed to download model weights: {e}")
        st.stop()

# Adjust model architecture to match the saved weights
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Include if the model was trained with dropout
    nn.Linear(model.fc.in_features, 5)
)

# Load the model weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define image transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("Lung Disease Detection")
st.write("Upload a chest X-ray image to classify lung disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_image = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        pred_class = torch.argmax(output, dim=1).item()

    st.write(f"Predicted Class: **{['COVID', 'Normal', 'Pneumonia', 'Pneumothorax', 'Tuberculosis'][pred_class]}**")
