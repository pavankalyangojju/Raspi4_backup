import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
import time

# Setup GPIO for Servo
SERVO_PIN = 21
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM
servo.start(0)

# Load recognizer and Haar cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = int(0.1 * cam.get(3))
minH = int(0.1 * cam.get(4))

def rotate_servo():
    print("[INFO] Rotating servo...")
    servo.ChangeDutyCycle(7.5)  # 90 degrees (open)
    time.sleep(1)
    servo.ChangeDutyCycle(0)
    time.sleep(2)
    servo.ChangeDutyCycle(2.5)  # 0 degrees (close)
    time.sleep(1)
    servo.ChangeDutyCycle(0)
    print("[INFO] Servo returned to original position")

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to capture image")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(minW, minH))

        access_granted = False

        for (x, y, w, h) in faces:
            id_predicted, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 30:
                label = f"User {id_predicted} ({round(confidence)}%)"
                access_granted = True
            else:
                label = f"Unknown ({round(confidence)}%)"

            # Draw box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if access_granted else (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if access_granted:
                rotate_servo()
                break

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(10) & 0xFF == 27 or access_granted:
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    print("[INFO] Cleaning up...")
    cam.release()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()
