import os
import cv2

# -------------------- CONFIG --------------------
DATASET_DIR = "dataset"  # Dataset folder to save captured images
IMG_SIZE = 160           # Resize captured face image to match model input
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -------------------- CREATE DATASET DIRECTORY --------------------
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# -------------------- CAPTURE SINGLE FACE IMAGE --------------------
def capture_image(person_name):
    # Create a directory for the person if it doesn't exist
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    # Start the video capture
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting to capture image for", person_name)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = frame[y:y+h, x:x+w]
            
            # Resize face to match the model's expected input
            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

            # Show the live feed with rectangle around the face
            cv2.imshow("Capture Face", frame)

        # Wait for 'c' to capture the image
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Press 'c' to capture the image
            if len(faces) > 0:
                img_path = os.path.join(person_dir, f"{person_name}.jpg")
                cv2.imwrite(img_path, face_resized)
                print(f"[INFO] Saved {img_path}")
                break  # Exit after saving the image

        # Exit the loop if the user presses the 'q' key
        elif key == ord('q'):
            print("[INFO] Exiting without capturing image.")
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------- MAIN --------------------
def main():
    # Get the name of the person for whom image is being captured
    person_name = input("[INFO] Enter the name of the person: ")

    # Capture and save a single image
    capture_image(person_name)

if __name__ == "__main__":
    main()
