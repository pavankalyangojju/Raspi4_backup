'''
Real-Time Face Registration with RFID Integration and CSV Logging
'''

import cv2
import os
import pandas as pd
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522

# Setup RFID Reader
GPIO.setwarnings(False)
reader = SimpleMFRC522()

# Initialize Camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # width
cam.set(4, 480)  # height

# Load Haar Cascade for face detection
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ensure CSV file exists
csv_file = 'user_data.csv'
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=['RFID_UID', 'Name'])
    df.to_csv(csv_file, index=False)

# Scan RFID Card
print("\n[INFO] Please scan your RFID card to begin...")
try:
    rfid_id, rfid_text = reader.read()
    print(f"\n[INFO] RFID ID: {rfid_id}")
except Exception as e:
    print(f"[ERROR] RFID Read Error: {e}")
    GPIO.cleanup()
    exit()
    
# Get User Name
name = input("Enter your name: ")

# Create folder named after RFID if not exists
rfid_folder = os.path.join('dataset', str(rfid_id))
os.makedirs(rfid_folder, exist_ok=True)

# Append RFID and Name to CSV
user_data = pd.DataFrame([[rfid_id, name]], columns=['RFID_UID', 'Name'])
user_data.to_csv(csv_file, mode='a', header=False, index=False)

print("\n[INFO] Initializing face capture. Look at the camera...")

count = 0
while True:
    ret, img = cam.read()
    img = cv2.flip(img, -1)  # flip image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Save face image into the RFID-specific folder
        filename = os.path.join(rfid_folder, f"{count}.jpg")
        cv2.imwrite(filename, gray[y:y+h, x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:
        break

print("\n[INFO] Exiting Program and cleaning up...")
cam.release()
cv2.destroyAllWindows()
GPIO.cleanup()
