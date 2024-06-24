import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

st.title("Image Classification with ViT")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_label = model.config.id2label[predicted_class_idx]

    st.write(f"Predicted class: {predicted_class_label}")


if __name__ == '__main__':
    st.write("Upload an image to classify")
