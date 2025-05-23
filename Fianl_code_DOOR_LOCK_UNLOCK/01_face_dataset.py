'''
Real-Time Face Registration with RFID Integration,
CSV Logging, and Centralized Face Storage (No RFID folders)
'''

import cv2
import os
import pandas as pd
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522

# --- GPIO Setup ---
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
LED_PIN = 18
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

# --- RFID Reader Setup ---
reader = SimpleMFRC522()

# --- Camera Setup ---
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# --- Haar Cascade ---
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# --- Centralized Dataset Folder ---
dataset_folder = 'faces'
os.makedirs(dataset_folder, exist_ok=True)

# --- CSV for User Info ---
csv_file = 'user_data.csv'
if not os.path.exists(csv_file):
    pd.DataFrame(columns=['RFID_UID', 'Name']).to_csv(csv_file, index=False)

try:
    # --- Scan RFID ---
    print("\n[INFO] Please scan your RFID card to begin...")
    rfid_id, _ = reader.read()
    print(f"[INFO] RFID ID: {rfid_id}")

    # --- Input Name ---
    name = input("Enter your name: ").strip().replace(" ", "_")

    # --- Append to CSV only if not already present ---
    existing_data = pd.read_csv(csv_file)
    if not ((existing_data['RFID_UID'] == rfid_id) & (existing_data['Name'] == name)).any():
        new_row = pd.DataFrame([[rfid_id, name]], columns=['RFID_UID', 'Name'])
        new_row.to_csv(csv_file, mode='a', header=False, index=False)
        print(f"[INFO] Registered: {name} with RFID {rfid_id}")
    else:
        print(f"[INFO] User {name} already registered with RFID {rfid_id}")

    print("[INFO] Starting face capture...")

    count = 0
    GPIO.output(LED_PIN, GPIO.HIGH)

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            filename = os.path.join(dataset_folder, f"{rfid_id}_{name}_{count}.jpg")
            cv2.imwrite(filename, gray[y:y+h, x:x+w])
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27 or count >= 30:  # ESC or 30 images
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user. Cleaning up...")

except Exception as e:
    print(f"[ERROR] {e}")

finally:
    GPIO.output(LED_PIN, GPIO.LOW)
    cam.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("[INFO] Program exited safely.")
