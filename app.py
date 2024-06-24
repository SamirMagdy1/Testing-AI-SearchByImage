from flask import Flask, request, jsonify
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import io

app = Flask(__name__)

# Load the model and processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess the image and make prediction
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

        predicted_class_idx = logits.argmax(-1).item()
        predicted_class_label = model.config.id2label[predicted_class_idx]

        return jsonify({"predicted_class": predicted_class_label}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
