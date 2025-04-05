import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd

# === Paths ===
dataset_path = 'faces'
trainer_path = 'trainer'
labels_csv = os.path.join(trainer_path, 'labels.csv')
trainer_file = os.path.join(trainer_path, 'trainer.yml')  # <-- changed filename to reflect LBPH

if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

# === Recognizer and Detector ===
recognizer = cv2.face.LBPHFaceRecognizer_create()  # <-- LBPH recognizer
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def extract_name(filename):
    try:
        return filename.split('_')[1]
    except:
        return None

def getImagesAndLabels(path):
    face_samples = []
    labels = []
    label_map = {}
    current_id = 0

    for image_file in sorted(os.listdir(path)):
        if not image_file.lower().endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(path, image_file)
        name = extract_name(image_file)
        if not name:
            continue

        if name not in label_map:
            label_map[name] = current_id
            current_id += 1

        try:
            pil_img = Image.open(image_path).convert('L')
        except:
            continue

        img_numpy = np.array(pil_img, 'uint8')
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(cv2.resize(img_numpy[y:y+h, x:x+w], (200, 200)))
            labels.append(label_map[name])

    return face_samples, labels, label_map

print("[INFO] Training LBPH Face Recognizer...")
faces, labels, label_map = getImagesAndLabels(dataset_path)

if len(faces) == 0:
    print("[ERROR] No faces found.")
    exit()

recognizer.train(faces, np.array(labels))
recognizer.write(trainer_file)

# Save label mapping
reverse_map = {v: k for k, v in label_map.items()}
df = pd.DataFrame([{"ID": k, "Name": v} for k, v in reverse_map.items()])
df.to_csv(labels_csv, index=False)

print(f"[INFO] Trained {len(reverse_map)} identities.")
print(f"[INFO] Saved LBPH model to: {trainer_file}")
