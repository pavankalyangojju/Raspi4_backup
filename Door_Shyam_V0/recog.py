
import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO

# -------------------- GPIO SETUP --------------------
BUZZER_PIN = 17
GREEN_LED_PIN = 27
RED_LED_PIN = 22
SERVO_PIN = 18
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

# -------------------- CONFIG --------------------
DATASET_DIR = "dataset"
TFLITE_MODEL = "facenet.tflite"
IMG_SIZE = 160
SIMILARITY_THRESHOLD = 0.7

GPIO.output(BUZZER_PIN, GPIO.LOW)
GPIO.output(GREEN_LED_PIN, GPIO.LOW)
GPIO.output(RED_LED_PIN, GPIO.LOW)

# === Servo Setup ===
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

# === RFID Reader ===
reader = SimpleMFRC522()

# -------------------- LOAD MODEL --------------------
print("[INFO] Loading TFLite FaceNet model...")
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------- LOAD HAAR CASCADE --------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -------------------- FUNCTIONS --------------------
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)

    expected_shape = input_details[0]['shape']
    if img.shape != tuple(expected_shape):
        print(f"[ERROR] Shape mismatch: got {img.shape}, expected {expected_shape}")
        return None

    if input_details[0]['dtype'] == np.uint8:
        img = img.astype(np.uint8)
    else:
        img = img.astype('float32') / 255.0

    return img

def get_embedding(face_img):
    face = preprocess(face_img)
    if face is None:
        raise ValueError("Preprocessing failed.")
    interpreter.set_tensor(input_details[0]['index'], face)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize(embedding, known_embeddings):
    best_match = None
    highest_similarity = -1

    for name, ref_emb in known_embeddings.items():
        similarity = cosine_similarity(embedding, ref_emb)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = name

    if highest_similarity >= SIMILARITY_THRESHOLD:
        global flag
        flag = True
        return best_match
    else:
        return "Unknown"

# -------------------- LOAD DATASET --------------------
print("[INFO] Loading and processing dataset images...")
known_faces = {}

for person in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_dir):
        continue
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print(f"[WARN] No face in {img_path}")
            continue

        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        emb = get_embedding(face)
        known_faces[person] = emb
        print(f"[OK] Embedding generated for {person}")
        break  # Use only one image per person

# === Load RFID mapping ===
try:
    with open("rfid_map.json", "r") as f:
        rfid_map = json.load(f)
except FileNotFoundError:
    print("[ERROR] rfid_map.json not found!")
    rfid_map = {}

if not known_faces:
    print("[ERROR] No embeddings found.")
    exit()

# === CSV Setup ===
attendance_log_file = "/home/pi/Desktop/AIDS/attendance_log.csv"
if not os.path.exists(attendance_log_file):
    with open(attendance_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "RFID", "Date", "Time"])

# === Restart program ===
def restart_program():
    print("[INFO] Restarting program...")
    time.sleep(1)
    os.execv(sys.executable, ['python3'] + sys.argv)
    
# === Servo control function ===
def rotate_servo():
    print("[INFO] Rotating servo & activating relays...")
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

    GPIO.output(LIGHT1_PIN, GPIO.HIGH)
    GPIO.output(LIGHT2_PIN, GPIO.HIGH)

    servo.ChangeDutyCycle(7.5)
    time.sleep(1)
    servo.ChangeDutyCycle(0)
    time.sleep(2)
    print("[INFO] Servo returned. Door unlocked.")

# === Telegram Bot Setup ===
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
last_update_id = 0

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

                reply = "Invalid command."
                if message == "/door_open":
                    reply = "Opening door..."
                    threading.Thread(target=rotate_servo).start()
                elif message == "/light1_on":
                    GPIO.output(LIGHT1_PIN, GPIO.HIGH)
                    reply = "LIGHT1 ON"
                elif message == "/light1_off":
                    GPIO.output(LIGHT1_PIN, GPIO.LOW)
                    reply = "LIGHT1 OFF"
                elif message == "/light2_on":
                    GPIO.output(LIGHT2_PIN, GPIO.HIGH)
                    reply = "LIGHT2 ON"
                elif message == "/light2_off":
                    GPIO.output(LIGHT2_PIN, GPIO.LOW)
                    reply = "LIGHT2 OFF"
                elif message == "/fan1_on":
                    GPIO.output(FAN1_PIN, GPIO.HIGH)
                    reply = "FAN1 ON"
                elif message == "/fan1_off":
                    GPIO.output(FAN1_PIN, GPIO.LOW)
                    reply = "FAN1 OFF"
                elif message == "/fan2_on":
                    GPIO.output(FAN2_PIN, GPIO.HIGH)
                    reply = "FAN2 ON"
                elif message == "/fan2_off":
                    GPIO.output(FAN2_PIN, GPIO.LOW)
                    reply = "FAN2 OFF"
                elif message == "/all_on":
                    for pin in [LIGHT1_PIN, LIGHT2_PIN, FAN1_PIN, FAN2_PIN]:
                        GPIO.output(pin, GPIO.HIGH)
                    reply = "ALL ON"
                elif message == "/all_off":
                    for pin in [LIGHT1_PIN, LIGHT2_PIN, FAN1_PIN, FAN2_PIN]:
                        GPIO.output(pin, GPIO.LOW)
                    reply = "ALL OFF"

                requests.post(f"{BASE_URL}/sendMessage", data={"chat_id": chat_id, "text": reply})
        except Exception as e:
            print(f"[ERROR] Telegram: {e}")
        time.sleep(2)

threading.Thread(target=telegram_listener, daemon=True).start()

# -------------------- REAL-TIME RECOGNITION --------------------
print("[INFO] Starting face recognition. Press 'q' to quit.")
cap = cv2.VideoCapture(0)
flag = False

# === Main Logic ===
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible!")
        restart_program()

    print("[INFO] Ready for attendance")

    try:
        while True:
            GPIO.output(GREEN_LED_PIN, GPIO.LOW)
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
                print("[WARNING] Unknown RFID")
                continue

            print(f"[INFO] RFID matched with {matched_user}, waiting for face...")
            c = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to capture frame.")
                    cap.release()
                    cv2.destroyAllWindows()
                    restart_program()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                if len(faces) == 0:
                    print("[WARNING] No face detected")
                    continue

                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                embedding = get_embedding(face_img)
                known_embedding = known_faces.get(matched_user)

                if known_embedding is None:
                    print("[ERROR] Face not found in dataset")
                    break

                similarity = cosine_similarity(embedding, known_embedding)
                print(f"[DEBUG] Similarity: {similarity:.4f}")

                if similarity > 0.8:
                    print(f"[SUCCESS] {matched_user} - Match: {similarity:.2f}")
                    current_time = datetime.now()
                    date = current_time.strftime("%Y-%m-%d")
                    time_str = current_time.strftime("%H:%M:%S")
                    with open(attendance_log_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([matched_user, scanned_uid, date, time_str])
                    GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
                    time.sleep(2)
                    rotate_servo()
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                else:
                    c += 1
                    if c > 15:
                        print("[WARNING] Face not matched")
                        GPIO.output(BUZZER_PIN, GPIO.HIGH)
                        GPIO.output(RED_LED_PIN, GPIO.HIGH)
                        time.sleep(3)
                        cap.release()
                        cv2.destroyAllWindows()
                        restart_program()

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        if cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()

# === Run Main Loop with Restart Capability ===
while True:
    main()
