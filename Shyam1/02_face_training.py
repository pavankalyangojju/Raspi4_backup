import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
dataset_path = 'dataset'
trainer_path = 'trainer'
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to get images and labels
def getImagesAndLabels(path):
    face_samples = []
    ids = []

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Folder name is RFID ID
        try:
            id = int(folder_name)
        except ValueError:
            print(f"[WARNING] Skipping folder with invalid RFID: {folder_name}")
            continue

        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            try:
                pil_img = Image.open(image_path).convert('L')  # Grayscale
            except Exception as e:
                print(f"[WARNING] Skipping image {image_path}: {e}")
                continue

            img_numpy = np.array(pil_img, 'uint8')
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)

    return face_samples, ids

print("\n[INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(dataset_path)
recognizer.train(faces, np.array(ids))

# Save the trained model
recognizer.write(os.path.join(trainer_path, 'trainer.yml'))

print(f"\n[INFO] {len(np.unique(ids))} unique faces trained. Exiting Program.")
