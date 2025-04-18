import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import json
import os

reader = SimpleMFRC522()
json_file = 'rfid_map.json'

try:
    print("Place your RFID tag near the reader...")
    id, _ = reader.read()  # We're not using the text from the tag
    name = input("Enter name for this RFID: ")

    print(f"ID: {id}")
    print(f"Name: {name}")

    # Load existing JSON data if it exists
    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
    else:
        data = []

    # Append new record
    data.append({"id": id, "name": name})

    # Save back to JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)  # ? fixed this line

    print("Data saved to rfid_map.json")

finally:
    GPIO.cleanup()
