import os
import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
MODEL_PATH = 'resnet50_lung_model.pth'
num_classes = 5  # Set based on your trained model

# Load the pre-trained ResNet50 model and modify the final layer
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Load model weights with error handling
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        st.success("Model weights loaded successfully.")
    except RuntimeError as e:
        st.error(f"Error loading model weights: {e}")
else:
    st.error("Model weights not found. Please ensure the model is trained and saved.")

# Define data transformation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

# Grad-CAM function
def generate_gradcam(model, img, target_class, final_conv_layer='layer4'):
    model.eval()
    
    # Prepare the image for Grad-CAM
    img = img.unsqueeze(0).to(device)
    img.requires_grad = True

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks to capture gradients and activations
    for name, module in model.named_modules():
        if name == final_conv_layer:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)

    # Forward pass
    output = model(img)
    pred_score = output[0, target_class]

    # Backward pass
    model.zero_grad()
    pred_score.backward()

    # Extract gradients and activations
    gradients = gradients[0].cpu().data.numpy()
    activations = activations[0].cpu().data.numpy()

    # Compute Grad-CAM
    weights = np.mean(gradients, axis=(2, 3))
    cam = np.zeros(activations.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        cam += w * activations[0, i, :, :]

    # Normalize and resize the CAM
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize between 0 and 1

    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # Convert BGR to RGB

    # Convert the input image to NumPy array for overlay
    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # Overlay heatmap on the original image
    overlay = 0.4 * heatmap + 0.6 * img_np  # Adjust overlay intensity
    overlay = overlay / overlay.max()

    return overlay

# Streamlit app UI
st.title("Lung Disease Detection using Grad-CAM")
st.write("Upload an image to visualize the Grad-CAM overlay.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Transform the image for model input
    input_image = test_transform(image).to(device)

    # Model inference
    with torch.no_grad():
        output = model(input_image.unsqueeze(0))
        pred_class = torch.argmax(output, dim=1).item()

    st.write(f"Predicted Class: {pred_class}")

    # Generate Grad-CAM visualization
    gradcam_overlay = generate_gradcam(model, input_image, pred_class)
    
    # Plot the Grad-CAM result
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(gradcam_overlay)
    ax.axis('off')
    st.pyplot(fig)

    st.write("Grad-CAM visualization generated successfully!")
