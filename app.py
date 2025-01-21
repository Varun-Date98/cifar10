import torch
from PIL import Image
import streamlit as st
from model import TinyVGG
import torchvision.transforms as transforms


# Defining variables
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Loading the model
model = TinyVGG(input_shape=3, num_classes=10)
model.load_state_dict(torch.load("./saved_models/TinyVGG.pth"))

# Setting the model in evaluation mode
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Streamlit app
st.title("CIFAR-10 Image Classification")
st.write("Upload an image, the model will predict its class! (NOTE: Model is trained on CIFAR-10 dataset)")
st.write(f"CIFAR-10 classes: {classes}")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict the class
    with st.spinner("Classifying..."):
        input_tensor = transform(image).unsqueeze(0)
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        max_prob, predicted_idx = torch.max(probabilities, 1)
        max_prob = max_prob.item()
        predicted_class = classes[predicted_idx.item()]

        # Displaying results
        probabilities_list = [f"{p:.2f}" for p in probabilities.tolist()[0]]
        prediction_text = {c: p for c, p in zip(classes, probabilities_list)}
        st.success(f"Prediction: **{predicted_class}** (Confidence: {max_prob:.2f})")
        st.write(f"Prediction Probabilities: {prediction_text}")
