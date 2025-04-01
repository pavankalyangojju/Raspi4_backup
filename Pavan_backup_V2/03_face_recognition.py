# -*- coding: utf-8 -*-
'''
Real Time Face Recognition with RFID Folder-Based Matching, Telegram Alerts, API Logging, and LCD/Buzzer Feedback
'''

import cv2
import os
import numpy as np
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import time
from PIL import Image
import smbus2
import requests
from datetime import datetime
import tempfile

# GPIO Setup
BUZZER_PIN = 17
GREEN_LED_PIN = 26
RED_LED_PIN = 19
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)
GPIO.output(GREEN_LED_PIN, GPIO.LOW)
GPIO.output(RED_LED_PIN, GPIO.LOW)

# LCD Setup
LCD_ADDR = 0x27
LCD_WIDTH = 16
LCD_CHR = 1
LCD_CMD = 0
LCD_BACKLIGHT = 0x08
ENABLE = 0b00000100
LINE_1 = 0x80
LINE_2 = 0xC0
bus = smbus2.SMBus(1)

# Telegram Config
BOT_TOKEN = "8129064480:AAFZZjw7UTUrPgwUW33xu_B51MyJPg3WneY"
CHAT_ID = "1367693706"

# API Endpoint
API_URL = "http://localhost:5000/attendance"

def lcd_byte(bits, mode):
    high_bits = mode | (bits & 0xF0) | LCD_BACKLIGHT
    low_bits = mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT
    bus.write_byte(LCD_ADDR, high_bits)
    lcd_toggle_enable(high_bits)
    bus.write_byte(LCD_ADDR, low_bits)
    lcd_toggle_enable(low_bits)

def lcd_toggle_enable(bits):
    time.sleep(0.0005)
    bus.write_byte(LCD_ADDR, bits | ENABLE)
    time.sleep(0.0005)
    bus.write_byte(LCD_ADDR, bits & ~ENABLE)
    time.sleep(0.0005)

def lcd_init():
    lcd_byte(0x33, LCD_CMD)
    lcd_byte(0x32, LCD_CMD)
    lcd_byte(0x06, LCD_CMD)
    lcd_byte(0x0C, LCD_CMD)
    lcd_byte(0x28, LCD_CMD)
    lcd_byte(0x01, LCD_CMD)
    time.sleep(0.005)

def lcd_display(message, line):
    lcd_byte(line, LCD_CMD)
    message = message.ljust(LCD_WIDTH, ' ')
    for char in message:
        lcd_byte(ord(char), LCD_CHR)

def send_telegram_photo(img, caption):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as temp:
            cv2.imwrite(temp.name, img)
            with open(temp.name, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': CHAT_ID, 'caption': caption}
                requests.post(url, files=files, data=data)
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram photo: {e}")

def send_attendance_api(name, rfid, timestamp):
    payload = {
        "name": name,
        "rfid": rfid,
        "datetime": timestamp
    }
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            print("[INFO] Attendance logged via API")
        else:
            print(f"[WARNING] API Error: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] API send failed: {e}")

lcd_init()
lcd_display("Welcome to", LINE_1)
lcd_display("AttendanceSystem", LINE_2)
time.sleep(2)

reader = SimpleMFRC522()
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

try:
    while True:
        lcd_display("Scan your", LINE_1)
        lcd_display("RFID Card...", LINE_2)

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)

        print("\n[INFO] Please scan your RFID card...")
        try:
            rfid_id, rfid_text = reader.read()
            rfid_id = str(rfid_id)
            print(f"[INFO] RFID Scanned: {rfid_id}")
            lcd_display("RFID Found", LINE_1)
            lcd_display("Processing...", LINE_2)
        except Exception as e:
            print(f"[ERROR] RFID Read Failed: {e}")
            lcd_display("RFID Read Error", LINE_1)
            lcd_display("Please Retry", LINE_2)
            GPIO.cleanup()
            break

        image_folder = os.path.join("dataset", rfid_id)
        if not os.path.exists(image_folder):
            print(f"[ERROR] No dataset folder found for RFID {rfid_id}")
            lcd_display("No Data Found", LINE_1)
            lcd_display("Access Denied", LINE_2)
            GPIO.cleanup()
            break

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
                    ids.append(1)
            return face_samples, ids

        print("[INFO] Training model from RFID-specific folder...")
        lcd_display("Training Face", LINE_1)
        lcd_display("Please Wait...", LINE_2)
        faces, ids = get_images_and_labels(image_folder)
        recognizer.train(faces, np.array(ids))

        print("[INFO] Model trained. Look at the camera...")
        lcd_display("Look at Camera", LINE_1)
        lcd_display("Verifying...", LINE_2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        matched = False

        while True:
            ret, img = cam.read()
            if not ret or img is None:
                print("[ERROR] Failed to read from camera")
                continue

            img = cv2.flip(img, -1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                id_pred, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 40:
                    name_file = os.path.join(image_folder, "name.txt")
                    if os.path.exists(name_file):
                        with open(name_file, "r") as f:
                            person_name = f.read().strip()
                    else:
                        person_name = "Matched"

                    print(f"[INFO] Face matched - {person_name} - Attendance Taken")
                    lcd_display("Your Attendance", LINE_1)
                    lcd_display(f"Welcome {person_name[:9]}", LINE_2)
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                    GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
                    time.sleep(0.2)
                    GPIO.output(BUZZER_PIN, GPIO.LOW)
                    time.sleep(3)
                    GPIO.output(GREEN_LED_PIN, GPIO.LOW)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    caption = f"Attendance Taken\nName: {person_name}\nRFID: {rfid_id}\nTime: {timestamp}"
                    send_telegram_photo(img, caption)
                    send_attendance_api(person_name, rfid_id, timestamp)

                    matched = True
                    break
                else:
                    print("[WARNING] Unknown face detected - Triggering buzzer")
                    lcd_display("Unknown Face", LINE_1)
                    lcd_display("Access Denied", LINE_2)
                    for _ in range(2):
                        GPIO.output(BUZZER_PIN, GPIO.HIGH)
                        time.sleep(0.2)
                        GPIO.output(BUZZER_PIN, GPIO.LOW)
                        time.sleep(0.2)
                    GPIO.output(RED_LED_PIN, GPIO.HIGH)
                    time.sleep(3)
                    GPIO.output(RED_LED_PIN, GPIO.LOW)
                    lcd_display("Put Correct", LINE_1)
                    lcd_display("Face", LINE_2)

                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("camera", img)
            if matched or cv2.waitKey(1) & 0xFF == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
        time.sleep(3)

except KeyboardInterrupt:
    print("\n[INFO] Program interrupted. Exiting gracefully.")
    lcd_display("Welcome to", LINE_1)
    lcd_display("AttendanceSystem", LINE_2)
    time.sleep(2)
    GPIO.cleanup()
