# main_rfid_face_telegram.py
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

def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)

# GPIO Setup
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

# Load Recognizer
recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.read('trainer/trainer_fisher.yml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# User data
user_df = pd.read_csv("user_data.csv")
labels_df = pd.read_csv("trainer/labels.csv")
labels_df.columns = labels_df.columns.str.strip()
labels_df['ID'] = labels_df['ID'].astype(int)
labels_df['Name'] = labels_df['Name'].astype(str).str.strip()

# RFID
rfid_reader = SimpleMFRC522()

# Telegram
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
last_update_id = 0

def servo_from_telegram():
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    servo.ChangeDutyCycle(2.5)
    time.sleep(1)
    servo.ChangeDutyCycle(12.5)
    time.sleep(1)
    servo.ChangeDutyCycle(0)
    time.sleep(5)
    servo.ChangeDutyCycle(2.5)
    time.sleep(1)
    servo.ChangeDutyCycle(0)

def rotate_servo():
    GPIO.output(LIGHT1_PIN, GPIO.HIGH)
    GPIO.output(RELAY2_PIN, GPIO.HIGH)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    servo.ChangeDutyCycle(7.5)
    time.sleep(1)
    servo.ChangeDutyCycle(0)
    time.sleep(2)
    servo.ChangeDutyCycle(2.5)
    time.sleep(1)
    servo.ChangeDutyCycle(0)
    cam.release()
    cv2.destroyAllWindows()

def telegram_listener():
    global last_update_id
    while True:
        try:
            url = f"{BASE_URL}/getUpdates?offset={last_update_id + 1}&timeout=10"
            response = requests.get(url).json()
            for update in response.get("result", []):
                last_update_id = update["update_id"]
                message = update.get("message", {}).get("text", "").lower()
                chat_id = update["message"]["chat"]["id"]
                reply = "Invalid command."
                if message == "/light1_on":
                    GPIO.output(LIGHT1_PIN, GPIO.HIGH)
                    reply = "LIGHT1 IS ON"
                elif message == "/light1_off":
                    GPIO.output(LIGHT1_PIN, GPIO.LOW)
                    reply = "LIGHT1 IS OFF"
                elif message == "/light2_on":
                    GPIO.output(RELAY2_PIN, GPIO.HIGH)
                    reply = "LIGHT2 IS ON"
                elif message == "/light2_off":
                    GPIO.output(RELAY2_PIN, GPIO.LOW)
                    reply = "LIGHT2 IS OFF"
                elif message == "/fan1_on":
                    GPIO.output(RELAY3_PIN, GPIO.HIGH)
                    reply = "FAN1 IS ON"
                elif message == "/fan1_off":
                    GPIO.output(RELAY3_PIN, GPIO.LOW)
                    reply = "FAN1 IS OFF"
                elif message == "/fan2_on":
                    GPIO.output(RELAY4_PIN, GPIO.HIGH)
                    reply = "FAN2 IS ON"
                elif message == "/fan2_off":
                    GPIO.output(RELAY4_PIN, GPIO.LOW)
                    reply = "FAN2 IS OFF"
                elif message == "/all_on":
                    GPIO.output(LIGHT1_PIN, GPIO.HIGH)
                    GPIO.output(RELAY2_PIN, GPIO.HIGH)
                    GPIO.output(RELAY3_PIN, GPIO.HIGH)
                    GPIO.output(RELAY4_PIN, GPIO.HIGH)
                    reply = "ALL ARE ON"
                elif message == "/all_off":
                    GPIO.output(LIGHT1_PIN, GPIO.LOW)
                    GPIO.output(RELAY2_PIN, GPIO.LOW)
                    GPIO.output(RELAY3_PIN, GPIO.LOW)
                    GPIO.output(RELAY4_PIN, GPIO.LOW)
                    reply = "ALL ARE OFF"
                elif message == "/door_open":
                    reply = "Door opening..."
                    threading.Thread(target=servo_from_telegram).start()
                requests.post(f"{BASE_URL}/sendMessage", data={"chat_id": chat_id, "text": reply})
        except Exception as e:
            print(f"[ERROR] Telegram: {e}")
        time.sleep(2)

# Start telegram listener
threading.Thread(target=telegram_listener, daemon=True).start()

# Camera setup
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
minW = int(0.1 * cam.get(3))
minH = int(0.1 * cam.get(4))

try:
    while True:
        print("[INFO] Waiting for RFID scan...")
        rfid_id, _ = rfid_reader.read()
        rfid_id = str(rfid_id).strip()
        print(f"[INFO] Scanned RFID: {rfid_id}")

        user_row = user_df[user_df['RFID_UID'] == int(rfid_id)]
        if user_row.empty:
            print("[ERROR] RFID not registered.")
            continue

        user_name = user_row.iloc[0]['Name']
        print(f"[INFO] RFID belongs to: {user_name}")

        access_granted = False

        while True:
            ret, frame = cam.read()
            if not ret:
                restart_program()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 7, minSize=(minW, minH))

            for (x, y, w, h) in faces:
                roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                id_predicted, confidence = recognizer.predict(roi)
                matched_row = labels_df[labels_df['ID'] == id_predicted]
                print("conf ",confidence)

                if confidence < 40:  # Lower threshold for EigenFace
                    predicted_name = matched_row.iloc[0]['Name'] if not matched_row.empty else "Unknown"
                    if predicted_name == user_name:
                        access_granted = True
                        label = f"✔ {predicted_name} ({round(confidence)})"
                    else:
                        label = f"✖ Mismatch: RFID={user_name}, Face={predicted_name}"
                else:
                    label = f"Unknown ({round(confidence)})"

                color = (0, 255, 0) if access_granted else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if access_granted:
                    rotate_servo()
                    break

            cv2.imshow('Face Recognition', frame)
            if access_granted or cv2.waitKey(10) & 0xFF == 27:
                break

except KeyboardInterrupt:
    print("[INFO] Program terminated.")

finally:
    cam.release()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()
