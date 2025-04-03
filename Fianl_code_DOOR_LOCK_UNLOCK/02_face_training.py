import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd

# Paths
dataset_path = 'faces'
trainer_path = 'trainer'
labels_csv = os.path.join(trainer_path, 'labels.csv')
trainer_file = os.path.join(trainer_path, 'trainer.yml')

if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to extract name from filename like '1234_pavan_1.jpg'
def extract_name(filename):
    try:
        parts = filename.split('_')
        return parts[1]  # 'pavan'
    except:
        return None

def getImagesAndLabels(path):
    face_samples = []
    labels = []
    label_map = {}         # name -> id
    current_id = 0

    for image_file in os.listdir(path):
        if not image_file.lower().endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(path, image_file)
        name = extract_name(image_file)

        if not name:
            print(f"[WARNING] Skipping file (can't parse name): {image_file}")
            continue

        if name not in label_map:
            label_map[name] = current_id
            current_id += 1

        try:
            pil_img = Image.open(image_path).convert('L')  # Grayscale
        except Exception as e:
            print(f"[WARNING] Skipping image {image_path}: {e}")
            continue

        img_numpy = np.array(pil_img, 'uint8')
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            labels.append(label_map[name])

    return face_samples, labels, label_map

print("\n[INFO] Training faces. Please wait...")
faces, labels, label_map = getImagesAndLabels(dataset_path)
recognizer.train(faces, np.array(labels))
recognizer.write(trainer_file)

# Save label mapping (name ? id)
label_df = pd.DataFrame(list(label_map.items()), columns=['Name', 'ID'])
label_df.to_csv(labels_csv, index=False)

print(f"\n[INFO] {len(label_map)} unique names trained.")
print(f"[INFO] Model saved to {trainer_file}")
print(f"[INFO] Labels saved to {labels_csv}")
