import os
import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import gdown

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model weights path and URL
MODEL_PATH = 'resnet50_lung_model.pth'
weights_url = "https://drive.google.com/uc?id=1QWb3w9u2eYPqWFPjSiXSdysnEjvyVG-n"

# Download weights if not found
if not os.path.exists(MODEL_PATH):
    gdown.download(weights_url, MODEL_PATH, quiet=False)

# Load the ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 5)  # Adjust for 5 classes
model = model.to(device)

try:
    # Load model weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# Define image transformations (for inference)
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Resize to fixed size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ResNet
])

# Streamlit UI
st.title("Lung Disease Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(output, dim=1).item()

    # Define class names
    class_names = ['COVID', 'Normal', 'Pneumonia', 'Pneumothorax', 'Tuberculosis']
    st.write(f"Predicted Class: **{class_names[pred_class]}**")
    st.write(f"Probabilities: {probs.cpu().numpy()}")
