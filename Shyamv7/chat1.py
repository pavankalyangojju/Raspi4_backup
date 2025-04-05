import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
import time
import pandas as pd
from mfrc522 import SimpleMFRC522

# === GPIO Setup ===
SERVO_PIN = 21
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

# === Load Recognizer and Cascade ===
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# === Load user data ===
user_df = pd.read_csv("user_data.csv")

# === RFID Setup ===
rfid_reader = SimpleMFRC522()

# === Camera Setup ===
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
minW = int(0.1 * cam.get(3))
minH = int(0.1 * cam.get(4))

def rotate_servo():
    print("[INFO] Rotating servo...")
    servo.ChangeDutyCycle(7.5)
    time.sleep(1)
    servo.ChangeDutyCycle(0)
    time.sleep(2)
    servo.ChangeDutyCycle(2.5)
    time.sleep(1)
    servo.ChangeDutyCycle(0)
    print("[INFO] Servo returned to original position")

try:
    while True:
        print("[INFO] Waiting for RFID scan...")
        rfid_id, _ = rfid_reader.read()
        rfid_id = str(rfid_id).strip()
        print(f"[INFO] Scanned RFID: {rfid_id}")

        # Find matching row from CSV
        user_row = user_df[user_df['RFID_UID'] == int(rfid_id)]
        if user_row.empty:
            print("[ERROR] This RFID is not registered in user_data.csv.")
            continue

        user_name = user_row.iloc[0]['Name']
        print(f"[INFO] RFID belongs to: {user_name}")

        print("[INFO] Starting face recognition...")

        access_granted = False

        while True:
            ret, frame = cam.read()
            if not ret:
                print("[ERROR] Failed to capture image")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(minW, minH))

            for (x, y, w, h) in faces:
                id_predicted, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 30:
                    label = f"{user_name} ({round(confidence)}%)"
                    access_granted = True
                else:
                    label = f"Unknown ({round(confidence)}%)"

                color = (0, 255, 0) if access_granted else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                if access_granted:
                    print(f"[ACCESS GRANTED] {user_name}")
                    rotate_servo()
                    break

            cv2.imshow('Face Recognition', frame)

            if access_granted or cv2.waitKey(10) & 0xFF == 27:
                break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    print("[INFO] Cleaning up...")
    cam.release()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()
