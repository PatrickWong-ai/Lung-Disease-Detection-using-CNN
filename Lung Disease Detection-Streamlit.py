import os
import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# Set device to CPU for Streamlit Cloud deployment
device = torch.device("cpu")

# Model weights path and URL
MODEL_PATH = 'resnet50_lung_model.pth'
weights_url = "https://drive.google.com/file/d/1QWb3w9u2eYPqWFPjSiXSdysnEjvyVG-n/view?usp=sharing"  # Replace with actual link
num_classes = 5  # Set based on your trained model
class_names = ['COVID', 'Normal', 'Pneumonia', 'Pneumothorax', 'Tuberculosis']  # Update with your classes

# Download weights if not found
if not os.path.exists(MODEL_PATH):
    try:
        import urllib.request
        urllib.request.urlretrieve(weights_url, MODEL_PATH)
        st.success("Model weights downloaded successfully.")
    except Exception as e:
        st.error(f"Failed to download model weights: {e}")
        st.stop()

# Load the model and weights
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    st.success("Model weights loaded successfully.")
except RuntimeError as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# Define data transformation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit app UI
st.title("Lung Disease Detection")
st.write("Upload a chest X-ray image to classify lung disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Transform the image for model input
    input_image = test_transform(image).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        output = model(input_image)
        pred_class = torch.argmax(output, dim=1).item()

    # Display the prediction
    st.write(f"Predicted Class: **{class_names[pred_class]}**")

