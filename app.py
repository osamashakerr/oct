import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load class names
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Define VGG16 model
def get_model():
    vgg16 = models.vgg16_bn()
    num_features = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_features, len(class_names))
    vgg16.load_state_dict(torch.load('VGG16_v2-OCT_Retina_half_dataset.pt', map_location=torch.device('cpu')))
    vgg16.eval()
    return vgg16

# Define image transformations
def transform_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Predict the class of the uploaded image
def predict(image, model):
    image_tensor = transform_image(image)
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]  # Get probabilities
        _, predicted = torch.max(outputs, 1)

    # Combine class names with their corresponding probabilities
    prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return class_names[predicted.item()], prob_dict

# Streamlit interface
st.title("Classification of retinal damage from OCT Scans")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    model = get_model()
    prediction, probabilities = predict(image, model)

    st.write(f"Prediction: {prediction}")
    st.write("Probabilities:")
    st.write(probabilities)
