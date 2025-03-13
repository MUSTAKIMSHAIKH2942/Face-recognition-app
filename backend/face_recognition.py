import cv2
import pickle
import csv
from datetime import datetime
import os
import logging

class FaceRecognition:
    def __init__(self, model_path, label_path, attendance_file):
        self.model_path = model_path
        self.label_path = label_path
        self.attendance_file = attendance_file

        # Load trained model and labels
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(self.model_path)

        with open(self.label_path, "rb") as f:
            self.label_names = pickle.load(f)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return gray, faces

    def recognize_faces(self, gray_frame, faces, frame):
        recognized_faces = []
        for (x, y, w, h) in faces:
            face = gray_frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (200, 200))
            label_id, confidence = self.recognizer.predict(face_resized)
            person_name = self.label_names.get(label_id, "Unknown")

            if person_name != "Unknown" and confidence < 50.0:
                self.record_attendance(person_name)
            else:
                self.capture_unknown_face(frame)

            recognized_faces.append((x, y, w, h, person_name, confidence))
        return recognized_faces

    def record_attendance(self, name):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        with open(self.attendance_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([date_str, name, time_str])

    def capture_unknown_face(self, frame):
        unknown_folder = os.path.join("data", "Unknown_images")
        os.makedirs(unknown_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(unknown_folder, f"unknown_{timestamp}.jpg")
        cv2.imwrite(file_path, frame)