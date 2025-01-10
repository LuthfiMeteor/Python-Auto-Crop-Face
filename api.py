from flask import Flask, request, jsonify, send_from_directory
import cv2
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_images'
CROPPED_FOLDER = 'cropped_images'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

def detect_face(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        padding = int(min(w, h) * 0.5)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return (x, y, w, h)
    else:
        return None

def crop_image(image_path, coordinates):
    original_name = os.path.basename(image_path)
    output_path = os.path.join(CROPPED_FOLDER, original_name)

    img = Image.open(image_path)
    if coordinates:
        x, y, w, h = coordinates
        cropped_img = img.crop((x, y, x + w, y + h))
        cropped_img = cropped_img.convert("RGB")
        cropped_img.save(output_path, "JPEG")
    else:
        img = img.convert("RGB")
        img.save(output_path, "JPEG")

    return output_path

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    coordinates = detect_face(file_path)
    cropped_image_path = crop_image(file_path, coordinates)

    cropped_url = f"http://127.0.0.1:5000/cropped_images/{os.path.basename(cropped_image_path)}"

    if coordinates:
        return jsonify({
            "message": "Face detected and image cropped successfully",
            "cropped_image_url": cropped_url
        })
    else:
        return jsonify({
            "message": "No face detected. Original image saved",
            "image_url": cropped_url
        })


@app.route('/', methods=['GET'])
def home():
    return "<h1>Face Detection API & Auto Crop By Face</h1>"

@app.route('/cropped_images/<filename>', methods=['GET'])
def get_cropped_image(filename):
    return send_from_directory(CROPPED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
