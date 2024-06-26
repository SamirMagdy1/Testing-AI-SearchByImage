from flask import Flask, request, jsonify
from google.cloud import vision
import io

app = Flask(__name__)

# ظ‚ظ… ط¨ط¥ط¹ط¯ط§ط¯ ط¹ظ…ظٹظ„ Google Cloud Vision
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
    
    # Use io.BytesIO to handle image content
    image_stream = io.BytesIO(image_content)
    image = vision.Image(content=image_stream.read())
    
    # ط·ظ„ط¨ ط§ظ„ط¨ط­ط« ط¨ط§ط³طھط®ط¯ط§ظ… ط§ظ„طµظˆط±ط©
    response = client.label_detection(image=image)
    labels = response.label_annotations
    
    # ط§ط³طھط®ط±ط§ط¬ ط§ظ„ظˆطµظˆظپط§طھ ظˆط§ظ„طھطµظ†ظٹظپط§طھ
    descriptions = [label.description for label in labels]
    
    return jsonify({'labels': descriptions})

if __name__ == '__main__':
    app.run(debug=False)