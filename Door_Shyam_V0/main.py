import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os, json, time, csv, sys
from datetime import datetime
from mfrc522 import SimpleMFRC522
from smbus2 import SMBus
from RPLCD.i2c import CharLCD
from threading import Thread
from queue import Queue
import RPi.GPIO as GPIO

# === GPIO Setup ===
BUZZER_PIN = 17
GREEN_LED_PIN = 26
RED_LED_PIN = 19
SERVO_PIN = 21

# === GPIO Setup ===
LIGHT1_PIN = 26
LIGHT2_PIN = 19
FAN1_PIN = 13
FAN2_PIN = 6

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

GPIO.setup(LIGHT1_PIN, GPIO.OUT)
GPIO.output(LIGHT1_PIN, GPIO.LOW)
GPIO.setup(LIGHT2_PIN, GPIO.OUT)
GPIO.output(LIGHT2_PIN, GPIO.LOW)
GPIO.setup(FAN1_PIN, GPIO.OUT)
GPIO.output(FAN1_PIN, GPIO.LOW)
GPIO.setup(FAN2_PIN, GPIO.OUT)
GPIO.output(FAN2_PIN, GPIO.LOW)


GPIO.output(BUZZER_PIN, GPIO.LOW)
GPIO.output(GREEN_LED_PIN, GPIO.LOW)
GPIO.output(RED_LED_PIN, GPIO.LOW)


# === Servo Setup ===
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

# === LCD Setup ===
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2)
lcd_queue = Queue()

# === Telegram Bot Setup ===
BOT_TOKEN = "7538789175:AAEj2AtmpVUjQOXXvEb5Lg7h0u-9ETxpZG4"
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
last_update_id = 0

def lcd_worker():
    while True:
        message1, message2, delay = lcd_queue.get()
        lcd.clear()
        lcd.write_string(message1)
        if message2:
            lcd.cursor_pos = (1, 0)
            lcd.write_string(message2)
        if delay > 0:
            time.sleep(delay)
        lcd_queue.task_done()

lcd_thread = Thread(target=lcd_worker, daemon=True)
lcd_thread.start()

# === RFID Reader ===
reader = SimpleMFRC522()

# === Load TFLite model ===
interpreter = tflite.Interpreter(model_path="facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Face Detection ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Preprocessing ===
def preprocess_face(face_img):
    face = cv2.resize(face_img, (160, 160))
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    return np.expand_dims(face, axis=0)

def get_embedding(face_img):
    preprocessed = preprocess_face(face_img)
    interpreter.set_tensor(input_details[0]['index'], preprocessed)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])
    return embedding[0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === Load known faces and embeddings ===
def load_known_faces():
    known_embeddings = {}
    for name in os.listdir("dataset"):
        path = f"dataset/{name}/face.jpg"
        if os.path.exists(path):
            img = cv2.imread(path)
            embedding = get_embedding(img)
            known_embeddings[name] = embedding
    return known_embeddings

known_faces = load_known_faces()

# === Load RFID mapping ===
with open("rfid_map.json", "r") as f:
    rfid_map = json.load(f)

# === CSV Setup ===
attendance_log_file = "/home/pi/Desktop/shyam/attendance_log.csv"
if not os.path.exists(attendance_log_file):
    with open(attendance_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "RFID", "Date", "Time"])

# === Restart program ===
def restart_program():
    print("[INFO] Restarting program...")
    lcd_queue.put(("Restarting...", "", 2))
    time.sleep(2)
    os.execv(sys.executable, ['python3'] + sys.argv)

# === Servo control function ===
def rotate_servo():
    print("[INFO] Rotating servo & activating relays...")
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

    # Turn on LIGHT1 and RELAY2 when servo is triggered
    GPIO.output(LIGHT1_PIN, GPIO.HIGH)  # Turn on LIGHT1
    GPIO.output(LIGHT2_PIN, GPIO.HIGH)  # Turn on LIGHT2

    # Rotate servo to unlock position
    servo.ChangeDutyCycle(7.5)  # Servo to 90 degrees (unlock)
    time.sleep(1)

    # Reset servo back to closed position
    servo.ChangeDutyCycle(0)  # Reset to original position
    time.sleep(2)

    print("[INFO] Servo returned. Door unlocked.")


            
# === Start Telegram Listener ===
threading.Thread(target=telegram_listener, daemon=True).start()
# === Telegram Bot Listener ===
def telegram_listener():
    global last_update_id
    print("[INFO] Telegram listener started...")
    while True:
        try:
            url = f"{BASE_URL}/getUpdates?offset={last_update_id + 1}&timeout=10"
            response = requests.get(url).json()
            for update in response.get("result", []):
                last_update_id = update["update_id"]
                message = update.get("message", {}).get("text", "").lower()
                chat_id = update["message"]["chat"]["id"]
                print(f"[TELEGRAM] {message}")

                if message == "/light1_on":
                    GPIO.output(LIGHT1_PIN, GPIO.HIGH)
                    reply = "LIGHT1 IS ON"
                elif message == "/light1_off":
                    GPIO.output(LIGHT1_PIN, GPIO.LOW)
                    reply = "LIGHT1 IS OFF"
                elif message == "/light2_on":
                    GPIO.output(LIGHT2_PIN, GPIO.HIGH)
                    reply = "LIGHT2 IS ON"
                elif message == "/light2_off":
                    GPIO.output(LIGHT2_PIN, GPIO.LOW)
                    reply = "LIGHT2 IS OFF"
                elif message == "/fan1_on":
                    GPIO.output(FAN1_PIN, GPIO.HIGH)
                    reply = "FAN1 IS ON"
                elif message == "/fan1_off":
                    GPIO.output(FAN1_PIN, GPIO.LOW)
                    reply = "FAN1 IS OFF"
                elif message == "/fan2_on":
                    GPIO.output(FAN2_PIN, GPIO.HIGH)
                    reply = "FAN2 IS ON"
                elif message == "/fan2_off":
                    GPIO.output(FAN2_PIN, GPIO.LOW)
                    reply = "FAN2 IS OFF"
                elif message == "/all_on":
                    GPIO.output(LIGHT1_PIN, GPIO.HIGH)
                    GPIO.output(LIGHT2_PIN, GPIO.HIGH)
                    GPIO.output(FAN1_PIN, GPIO.HIGH)
                    GPIO.output(FAN2_PIN,GPIO.HIGH)
                    reply = "ALL ARE ON"
                elif message == "/all_off":
                    GPIO.output(LIGHT1_PIN, GPIO.LOW)
                    GPIO.output(LIGHT2_PIN, GPIO.LOW)
                    GPIO.output(FAN1_PIN, GPIO.LOW)
                    GPIO.output(FAN2_PIN, GPIO.LOW)
                    reply = "ALL ARE OFF"
                elif message == "/door_open":
                    reply = "Rotating servo from Telegram..."
                    threading.Thread(target=servo_from_telegram).start()
                else:
                    reply = "Send a valid command."

                requests.post(f"{BASE_URL}/sendMessage", data={"chat_id": chat_id, "text": reply})
        except Exception as e:
            print(f"[ERROR] Telegram: {e}")
        time.sleep(2)

# === Start Telegram Listener ===
threading.Thread(target=telegram_listener, daemon=True).start()

# === Main Logic ===
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible!")
        lcd_queue.put(("Camera Error", "", 2))
        restart_program()

    print("[INFO] Ready for attendance")

    try:
        while True:
            GPIO.output(GREEN_LED_PIN, GPIO.LOW)
            lcd_queue.put(("Scan RFID Card...", "", 0))
            print("Waiting for RFID...")
            id, text = reader.read()
            scanned_uid = str(id).strip()
            print("RFID scanned:", scanned_uid)

            matched_user = None
            for name, uid in rfid_map.items():
                if uid == scanned_uid:
                    matched_user = name
                    break

            if not matched_user:
                lcd_queue.put(("Unknown RFID", "", 2))
                continue

            lcd_queue.put(("Show your face", "", 0))
            print(f"[INFO] RFID matched with {matched_user}, waiting for face...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to capture frame.")
                    cap.release()
                    cv2.destroyAllWindows()
                    restart_program()
                cv2.imshow("Camera Feed", frame)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                if len(faces) == 0:
                    lcd_queue.put(("No face found", "", 1))
                    print("[WARNING] No face detected")
                    cv2.imshow("Camera Feed", frame)
                    continue

                # Use first detected face
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                

                embedding = get_embedding(face_img)
                known_embedding = known_faces.get(matched_user)

                if known_embedding is None:
                    lcd_queue.put(("Face Not Found", "", 2))
                    break

                similarity = cosine_similarity(embedding, known_embedding)
                print(f"[DEBUG] Similarity: {similarity:.4f}")

                if similarity > 0.8:
                    lcd_queue.put((matched_user, "Attendance Marked", 2))
                    print(f"[SUCCESS] {matched_user} - Match: {similarity:.2f}")

                    current_time = datetime.now()
                    date = current_time.strftime("%Y-%m-%d")
                    time_str = current_time.strftime("%H:%M:%S")

                    with open(attendance_log_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([matched_user, scanned_uid, date, time_str])

                    GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
                    time.sleep(2)
                    rotate_servo()  # Rotate servo to unlock door
                    cap.release()
                    cv2.destroyAllWindows()
                    return  # Restart loop for next person

                else:
                    lcd_queue.put(("Face Not Match", "", 1))
                    print("[WARNING] Face not matched")
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                    GPIO.output(RED_LED_PIN, GPIO.HIGH)
                    time.sleep(3)
                    cap.release()
                    cv2.destroyAllWindows()
                    restart_program()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        if cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()
            lcd.clear()

# === Run Main Loop with Restart Capability ===
while True:
    main()
