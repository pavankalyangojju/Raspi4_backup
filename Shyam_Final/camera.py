import cv2
import os

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ask for the user's name
name = input("Enter the name of the person: ").strip()
save_dir = os.path.join("dataset", name)

# Create directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access the camera.")
    exit()

print("[INFO] Press 's' to save the face, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        margin = 10
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = x + w + margin, y + h + margin
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        face_img = frame[y1:y2, x1:x2]

    cv2.imshow("Capture Face", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if len(faces) == 0:
            print("[WARNING] No face to save.")
        else:
            save_path = os.path.join(save_dir, "face.jpg")
            cv2.imwrite(save_path, face_img)
            print(f"[INFO] Face saved to {save_path}")
            break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
