import os
import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import gdown

# Set device to CPU
device = torch.device("cuda")

# Model weights path and URL
MODEL_PATH = 'resnet50_lung_model.pth'
weights_url = "https://drive.google.com/uc?id=1QWb3w9u2eYPqWFPjSiXSdysnEjvyVG-n"

# Download weights if not found
if not os.path.exists(MODEL_PATH):
    gdown.download(weights_url, MODEL_PATH, quiet=False)

# Load the model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 5)  # Adjust for 5 classes
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# Define image transformations
image_size = 224

transform = transforms.Compose([
    transforms.RandomRotation(5),  # Minimal rotation
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small translation
    transforms.Resize((image_size, image_size)),  # Fixed size
    transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),  # Random crop with scale limit
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Standard normalization for grayscale
])

# Streamlit UI
st.title("Lung Disease Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        st.write(f"Model output (logits): {output.cpu().numpy()}")  # Debugging line
        probs = torch.softmax(output, dim=1)
        st.write(f"Probabilities: {probs.cpu().numpy()}")  # Debugging line
        pred_class = torch.argmax(output, dim=1).item()

    class_names = ['COVID', 'Normal', 'Pneumonia', 'Pneumothorax', 'Tuberculosis']
    st.write(f"Predicted Class: **{class_names[pred_class]}**")
