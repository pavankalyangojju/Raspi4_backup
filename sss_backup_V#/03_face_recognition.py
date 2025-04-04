import cv2
import os
import numpy as np
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import time
from PIL import Image
import pandas as pd
import threading
import requests

# GPIO Setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

RELAY_PINS = {
    "relay1": 26,
    "relay2": 19,
    "relay3": 13,
    "relay4": 6,
}
SERVO_PIN = 17
GREEN_LED = 22
RED_LED = 27
BUZZER = 5

ALL_OUTPUTS = list(RELAY_PINS.values()) + [SERVO_PIN, GREEN_LED, RED_LED, BUZZER]
for pin in ALL_OUTPUTS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

# Telegram Bot
BOT_TOKEN = "7038070025:AAHOoUWmqVPvFmmITJKpbWVGcdwzLDmcVJI"
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
last_update_id = 0

def open_close():
    servo.ChangeDutyCycle(12.5)
    time.sleep(1)
    servo.ChangeDutyCycle(0)
    time.sleep(2)
    servo.ChangeDutyCycle(2.5)
    time.sleep(1)
    servo.ChangeDutyCycle(0)

def handle_telegram():
    global last_update_id
    print("[INFO] Telegram listener started...")
    while True:
        try:
            url = f"{BASE_URL}/getUpdates?offset={last_update_id + 1}&timeout=10"
            res = requests.get(url).json()

            for update in res.get("result", []):
                last_update_id = update["update_id"]
                msg = update.get("message", {}).get("text", "").lower()
                chat_id = update["message"]["chat"]["id"]
                print(f"[TELEGRAM] Received: {msg}")

                if msg == "/relayall_on":
                    for pin in RELAY_PINS.values():
                        GPIO.output(pin, GPIO.HIGH)
                    response = "All relays turned ON"
                    requests.post(f"{BASE_URL}/sendMessage", data={"chat_id": chat_id, "text": response})

                elif msg == "/relayall_off":
                    for pin in RELAY_PINS.values():
                        GPIO.output(pin, GPIO.LOW)
                    response = "All relays turned OFF"
                    requests.post(f"{BASE_URL}/sendMessage", data={"chat_id": chat_id, "text": response})

                elif msg.startswith("/relay"):
                    parts = msg.split("_")
                    if len(parts) == 2:
                        relay, state = parts
                        relay = relay.lstrip("/")
                        if relay in RELAY_PINS:
                            if state == "on":
                                GPIO.output(RELAY_PINS[relay], GPIO.HIGH)
                                response = f"{relay} turned ON"
                            elif state == "off":
                                GPIO.output(RELAY_PINS[relay], GPIO.LOW)
                                response = f"{relay} turned OFF"
                            else:
                                response = "Invalid state command."
                            requests.post(f"{BASE_URL}/sendMessage", data={"chat_id": chat_id, "text": response})
        except Exception as e:
            print(f"[ERROR] Telegram handler: {e}")
        time.sleep(2)

def matched_actions():
    GPIO.output(GREEN_LED, GPIO.HIGH)
    GPIO.output(BUZZER, GPIO.HIGH)
    time.sleep(2)  # Buzzer on for 2 seconds
    GPIO.output(BUZZER, GPIO.LOW)
    time.sleep(3)  # Remaining 3 seconds for LED
    GPIO.output(GREEN_LED, GPIO.LOW)

def unmatched_actions():
    GPIO.output(RED_LED, GPIO.HIGH)
    for _ in range(2):
        GPIO.output(BUZZER, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(BUZZER, GPIO.LOW)
        time.sleep(0.5)
    time.sleep(3)  # Remaining LED time
    GPIO.output(RED_LED, GPIO.LOW)

def recognize_and_act():
    reader = SimpleMFRC522()
    try:
        print("[INFO] Waiting for RFID...")
        rfid_id, _ = reader.read()
        rfid_id = str(rfid_id).strip()
        print(f"[INFO] RFID Scanned: {rfid_id}")
    except Exception as e:
        print(f"[ERROR] RFID failed: {e}")
        return

    dataset_path = os.path.join("dataset", rfid_id)
    if not os.path.exists(dataset_path):
        print(f"[ERROR] No dataset for RFID: {rfid_id}")
        return

    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    def load_training_data(path):
        faces, ids = [], []
        for file in os.listdir(path):
            if file.endswith(".jpg"):
                img = Image.open(os.path.join(path, file)).convert("L")
                np_img = np.array(img, 'uint8')
                for (x, y, w, h) in face_detector.detectMultiScale(np_img):
                    faces.append(np_img[y:y+h, x:x+w])
                    ids.append(1)
        return faces, ids

    print("[INFO] Training recognizer...")
    faces, ids = load_training_data(dataset_path)
    recognizer.train(faces, np.array(ids))

    try:
        df = pd.read_csv("user_data.csv")
        user_row = df[df["RFID_UID"] == int(rfid_id)]
        user_name = user_row.iloc[0]["Name"] if not user_row.empty else "Unknown"
    except:
        user_name = "Unknown"

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    matched = False
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            print("[ERROR] Camera failed to read")
            continue

        frame = cv2.flip(frame, -1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id_pred, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 40:
                print(f"[MATCH] {user_name} (Conf: {conf:.2f})")
                open_close()
                GPIO.output(RELAY_PINS["relay1"], GPIO.HIGH)
                GPIO.output(RELAY_PINS["relay2"], GPIO.HIGH)
                GPIO.output(RELAY_PINS["relay3"], GPIO.LOW)
                GPIO.output(RELAY_PINS["relay4"], GPIO.LOW)
                matched_actions()
                matched = True
                break
            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                unmatched_actions()

        cv2.imshow("Face Recognition", frame)
        if matched or cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

# Start the system
if __name__ == "__main__":
    threading.Thread(target=handle_telegram, daemon=True).start()
    while True:
        recognize_and_act()
