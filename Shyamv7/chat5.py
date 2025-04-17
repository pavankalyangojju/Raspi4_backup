import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# -------------------- CONFIG --------------------
DATASET_DIR = "dataset"
TFLITE_MODEL = "facenet.tflite"
IMG_SIZE = 160
SIMILARITY_THRESHOLD = 0.5  # Cosine similarity (closer to 1.0 is better)

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

    input_dtype = input_details[0]['dtype']

    if input_dtype == np.uint8:
        img = img.astype(np.uint8)
    else:  # float32
        img = img.astype('float32') / 255.0

    return np.expand_dims(img, axis=0)

def get_embedding(face_img):
    face = preprocess(face_img)
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

if not known_faces:
    print("[ERROR] No embeddings found.")
    exit()

# -------------------- REAL-TIME RECOGNITION --------------------
print("[INFO] Starting face recognition. Press 'q' to quit.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        try:
            emb = get_embedding(face_img)
            name = recognize(emb, known_faces)
        except Exception as e:
            name = "Error"
            print("[ERROR] Embedding failed:", e)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
