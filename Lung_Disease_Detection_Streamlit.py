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

# Download weights from Google Drive if not found locally
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights..."):
        gdown.download(weights_url, MODEL_PATH, quiet=False)

# Load ResNet50 architecture and replace the final layer
model = models.resnet50(weights=None)  # No pre-trained weights

# Adjust the fully connected layer to match the state dict
num_classes = 5  # Adjust for your use case
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),  # Add an intermediate layer if required
    nn.ReLU(),
    nn.Linear(512, num_classes)
)
model = model.to(device)

# Load custom weights
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # Set model to evaluation mode
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# Define image transformations for inference
image_size = 224
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.Resize((image_size, image_size)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])  # Normalization for ResNet
])

# Streamlit UI
st.title("Lung Disease Detection using ResNet50")
st.write("Upload a chest X-ray image to detect lung disease.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open the uploaded image and convert to grayscale
    image = Image.open(uploaded_file).convert("L")  # Load as grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_image = transform(image).unsqueeze(0).to(device)

    # Perform inference
with torch.no_grad():
    output = model(input_image)
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]  # Convert to NumPy array
    pred_class = torch.argmax(output, dim=1).item()

# Define class names
class_names = ['COVID', 'Normal', 'Pneumonia', 'Pneumothorax', 'Tuberculosis']

# Format probabilities as percentages
formatted_probs = [f"{p * 100:.2f}%" for p in probs]

# Display the results
st.write(f"Predicted Class: **{class_names[pred_class]}**")
st.write(f"Probabilities: {formatted_probs}")
