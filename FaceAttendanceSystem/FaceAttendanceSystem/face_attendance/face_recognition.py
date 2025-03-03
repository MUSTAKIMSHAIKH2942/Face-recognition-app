import cv2
import pickle
import csv
from datetime import datetime
import logging
import os

class FaceRecognition:
    def __init__(self, camera_position="Camera 1", log_callback=None):
        self.model_path = os.path.join("data", "models", "trained_model.yml")
        self.label_path = os.path.join("data", "models", "label_names.pkl")
        self.attendance_file = os.path.join("data", "attendance.csv")
        self.camera_position = camera_position
        self.log_callback = log_callback
        self.detected_persons = set()

        self.initialize_model()

    def initialize_model(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.label_path):
            raise FileNotFoundError("Trained model or label file not found. Train the model first.")

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(self.model_path)

        with open(self.label_path, "rb") as label_file:
            self.label_names = pickle.load(label_file)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Date", "Name", "Camera Position", "In Time", "Out Time"])

    def log_message(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return gray, faces

    def recognize_faces(self, gray_frame, faces):
        recognized_faces = []
        for (x, y, w, h) in faces:
            face = gray_frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (200, 200))
            label_id, confidence = self.recognizer.predict(face_resized)
            person_name = self.label_names.get(label_id, "Unknown")

            if person_name != "Unknown" and confidence < 50.0:
                self.record_attendance(person_name)
                if person_name not in self.detected_persons:
                    self.detected_persons.add(person_name)
                    self.log_message(f"Detected for the first time: {person_name}")

            recognized_faces.append((x, y, w, h, person_name, confidence))
        return recognized_faces

    def record_attendance(self, name):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        with open(self.attendance_file, "r") as file:
            rows = list(csv.reader(file))

        today_entries = [row for row in rows if row[0] == date_str and row[1] == name and row[2] == self.camera_position]

        if not today_entries:
            with open(self.attendance_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([date_str, name, self.camera_position, time_str, ""])
        else:
            for i, row in enumerate(rows):
                if row[0] == date_str and row[1] == name and row[2] == self.camera_position:
                    rows[i][4] = time_str
                    break
            with open(self.attendance_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(rows)

    def update_frame_with_recognition(self, frame):
        gray_frame, faces = self.detect_faces(frame)
        recognized_faces = self.recognize_faces(gray_frame, faces)

        for (x, y, w, h, person_name, confidence) in recognized_faces:
            color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{person_name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame