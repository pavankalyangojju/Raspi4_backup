import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
import time
import pandas as pd
import threading
import requests
from mfrc522 import SimpleMFRC522
import sys

# === Restart logic ===
def restart_program():
    print("[INFO] Restarting program...")
    python = sys.executable
    os.execl(python, python, *sys.argv)

# === GPIO Setup ===
SERVO_PIN = 21
BUZZER_PIN = 20
LIGHT1_PIN = 26
RELAY2_PIN = 19
RELAY3_PIN = 13
RELAY4_PIN = 6
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LIGHT1_PIN, GPIO.OUT)
GPIO.output(LIGHT1_PIN, GPIO.LOW)
GPIO.setup(RELAY2_PIN, GPIO.OUT)
GPIO.output(RELAY2_PIN, GPIO.LOW)
GPIO.setup(RELAY3_PIN, GPIO.OUT)
GPIO.output(RELAY3_PIN, GPIO.LOW)
GPIO.setup(RELAY4_PIN, GPIO.OUT)
GPIO.output(RELAY4_PIN, GPIO.LOW)

servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

# === Load Recognizer and Cascade ===
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# === Load user and label data ===
user_df = pd.read_csv("user_data.csv")  # Ensure columns: RFID_UID, Name
labels_df = pd.read_csv("trainer/labels.csv")  # Ensure columns: ID, Name

# === RFID Setup ===
rfid_reader = SimpleMFRC522()

# === Camera Setup ===
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
minW = int(0.1 * cam.get(3))
minH = int(0.1 * cam.get(4))

# === Main Loop ===
try:
    while True:
        print("[INFO] Waiting for RFID scan...")
        rfid_id, _ = rfid_reader.read()
        rfid_id = str(rfid_id).strip()
        print(f"[INFO] Scanned RFID: {rfid_id}")

        # Lookup name for RFID
        user_row = user_df[user_df['RFID_UID'] == int(rfid_id)]
        if user_row.empty:
            print("[ERROR] This RFID is not registered.")
            continue

        user_name = user_row.iloc[0]['Name']
        print(f"[INFO] RFID belongs to: {user_name}")

        access_granted = False

        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                print("[ERROR] Failed to capture image from camera")
                time.sleep(2)
                restart_program()

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except cv2.error as e:
                print(f"[ERROR] OpenCV Error: {e}")
                cam.release()
                cv2.destroyAllWindows()
                time.sleep(2)
                restart_program()

            faces = faceCascade.detectMultiScale(gray, 1.1, 7, minSize=(minW, minH))

            for (x, y, w, h) in faces:
                id_predicted, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                if confidence < 40:  # Higher confidence = lower number
                    matched_row = labels_df[labels_df['ID'] == id_predicted]
                    if not matched_row.empty:
                        predicted_name = matched_row.iloc[0]['Name']
                        print(f"[INFO] Face Recognized as: {predicted_name} (Confidence: {round(confidence)}%)")

                        # ✅ Match RFID name with Face Recognition name
                        if predicted_name == user_name:
                            access_granted = True
                            label = f"✅ {predicted_name} ({round(confidence)}%)"
                        else:
                            label = f"❌ Mismatch! RFID: {user_name}, Face: {predicted_name}"
                            print(f"[ALERT] Name Mismatch! Denying Access.")

                    else:
                        label = f"Unknown ({round(confidence)}%)"
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
