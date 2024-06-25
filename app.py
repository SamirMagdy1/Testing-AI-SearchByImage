from flask import Flask, request, jsonify
from google.cloud import vision
import io

app = Flask(__name__)

# قم بإعداد عميل Google Cloud Vision
client = vision.ImageAnnotatorClient()

@app.route('/')
def index():
    return 'Welcome to the Image Search API!'

@app.route('/search-by-image', methods=['POST'])
def search_by_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    image_content = image_file.read()
    
    image = vision.Image(content=image_content)
    
    # طلب البحث باستخدام الصورة
    response = client.label_detection(image=image)
    labels = response.label_annotations
    
    # استخراج الوصوفات والتصنيفات
    descriptions = [label.description for label in labels]
    
    return jsonify({'labels': descriptions})

if __name__ == '__main__':
    app.run(debug=False)