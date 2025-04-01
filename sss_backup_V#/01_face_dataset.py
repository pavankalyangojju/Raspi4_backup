'''
Real-Time Face Registration with RFID Integration, CSV Logging, and LED Indicator with CTRL+C Handling
'''

import cv2
import os
import pandas as pd
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522

# --- GPIO Setup ---
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

LED_PIN = 18  # GPIO pin connected to the LED
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

# --- RFID Reader Setup ---
reader = SimpleMFRC522()

# --- Camera Setup ---
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# --- Haar Cascade for Face Detection ---
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# --- CSV for User Info ---
csv_file = 'user_data.csv'
if not os.path.exists(csv_file):
    pd.DataFrame(columns=['RFID_UID', 'Name']).to_csv(csv_file, index=False)

try:
    # --- Read RFID ---
    print("\n[INFO] Please scan your RFID card to begin...")
    rfid_id, _ = reader.read()
    print(f"\n[INFO] RFID ID: {rfid_id}")

    # --- Get User Name ---
    name = input("Enter your name: ")

    # --- Create Folder for Dataset ---
    rfid_folder = os.path.join('dataset', str(rfid_id))
    os.makedirs(rfid_folder, exist_ok=True)

    # --- Append User Info to CSV ---
    user_data = pd.DataFrame([[rfid_id, name]], columns=['RFID_UID', 'Name'])
    user_data.to_csv(csv_file, mode='a', header=False, index=False)

    print("\n[INFO] Initializing face capture. Look at the camera...")

    count = 0
    GPIO.output(LED_PIN, GPIO.HIGH)  # Turn on LED while capturing

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            filename = os.path.join(rfid_folder, f"{count}.jpg")
            cv2.imwrite(filename, gray[y:y + h, x:x + w])
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27 or count >= 30:  # ESC or 30 images
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user. Cleaning up...")

except Exception as e:
    print(f"[ERROR] {e}")

finally:
    # Ensure LED and GPIO cleanup
    GPIO.output(LED_PIN, GPIO.LOW)
    cam.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("[INFO] Program exited safely.")
