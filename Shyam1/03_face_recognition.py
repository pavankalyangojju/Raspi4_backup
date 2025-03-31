'''
Real Time Face Recognition with RFID Folder-Based Matching + Servo Control
'''

import cv2
import os
import numpy as np
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import time
from PIL import Image
import pandas as pd

# Setup GPIO
RELAY_PIN = 26
SERVO_PIN = 17  # Servo connected to GPIO 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

GPIO.output(RELAY_PIN, GPIO.LOW)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM for servo
servo.start(0)  # Start with 0 duty cycle

reader = SimpleMFRC522()
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Start camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

print("\n[INFO] Please scan your RFID card...")
try:
    rfid_id, rfid_text = reader.read()
    rfid_id = str(rfid_id)
    print(f"[INFO] RFID Scanned: {rfid_id}")
except Exception as e:
    print(f"[ERROR] RFID Read Failed: {e}")
    GPIO.cleanup()
    exit()

# Check if dataset folder for this RFID exists
image_folder = os.path.join("dataset", rfid_id)
if not os.path.exists(image_folder):
    print(f"[ERROR] No dataset folder found for RFID {rfid_id}")
    GPIO.cleanup()
    exit()

# Prepare training data from that folder
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, 'uint8')
        faces = face_detector.detectMultiScale(img_np)
        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y+h, x:x+w])
            ids.append(1)  # Dummy label as we're only validating one user

    return face_samples, ids

print("[INFO] Training model from RFID-specific folder...")
faces, ids = get_images_and_labels(image_folder)
recognizer.train(faces, np.array(ids))

print("[INFO] Model trained. Look at the camera...")
font = cv2.FONT_HERSHEY_SIMPLEX

# Read user data from CSV
csv_file = 'user_data.csv'
user_data = pd.read_csv(csv_file)
user_row = user_data[user_data['RFID_UID'] == int(rfid_id)]
user_name = user_row.iloc[0]['Name']
print(user_name)

flag = False
while True:
    ret, img = cam.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id_pred, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face

        if confidence < 40:
            cv2.putText(img, user_name, (x+5, y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255,255,0), 1) 
            print("[INFO] Face matched with RFID folder. Attendance Taken.")
            flag = True
           
        else:
            cv2.putText(img, "Unknown Face", (x, y - 10), font, 1, (0, 0, 255), 2)
            cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow("camera", img)
    
    if flag:
        servo.ChangeDutyCycle(12.5)  
        time.sleep(1)  # Wait for servo to move
        servo.ChangeDutyCycle(0)  # Stop sending signal

        time.sleep(3)  # Hold for 3 seconds
        servo.ChangeDutyCycle(2.5)  
        time.sleep(1)  # Wait for servo to reset
        servo.ChangeDutyCycle(0)  # Stop sending signal
        
   
    if flag or cv2.waitKey(1) & 0xFF == 27:
        break
        
        

# Cleanup
cam.release()
cv2.destroyAllWindows()
GPIO.output(RELAY_PIN, GPIO.LOW)
servo.stop()
GPIO.cleanup()
