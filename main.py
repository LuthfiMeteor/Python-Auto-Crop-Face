import cv2
from PIL import Image
# import uuid
import os

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
    if coordinates:
        x, y, w, h = coordinates
        
        img = Image.open(image_path)
        cropped_img = img.crop((x, y, x + w, y + h))

        output_folder = "cropped_images"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        original_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, original_name)
        cropped_img = cropped_img.convert("RGB")
        cropped_img.save(output_path, "JPEG")
        
        print(f"Gambar berhasil dipotong dan disimpan sebagai '{output_path}'")
    else:
        output_folder = "cropped_images"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        original_name = os.path.basename(image_path)
        name, ext = os.path.splitext(original_name)
        output_path = os.path.join(output_folder, f"{name}_NC{ext}")
        img = Image.open(image_path)
        img = img.convert("RGB")
        img.save(output_path, "JPEG")
        
        print(f"Tidak ada wajah yang terdeteksi di gambar: {image_path}. Gambar dipindahkan ke '{output_path}'")

def process_folder(folder_path):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

    for file_name in os.listdir(folder_path):
        if file_name.endswith(supported_extensions):
            image_path = os.path.join(folder_path, file_name)
            print(f"Memproses gambar: {image_path}")
            
            coordinates = detect_face(image_path)
            crop_image(image_path, coordinates)

def main():
    folder_path = 'photo-id' 
    process_folder(folder_path)

if __name__ == "__main__":
    main()
