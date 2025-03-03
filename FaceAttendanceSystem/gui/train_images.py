from PyQt5.QtWidgets import * 
import cv2
import os
import pickle
import numpy as np
import logging

class TrainImagesScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train Images")
        self.setGeometry(200, 200, 800, 600)
        self.layout = QVBoxLayout(self)
        self.setup_ui()

    def setup_ui(self):
        title = QLabel("Train Images for Face Recognition")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        self.layout.addWidget(title)

        instructions = QLabel("Click the button below to start training the model with available images.")
        instructions.setStyleSheet("font-size: 14px; margin-bottom: 20px;")
        self.layout.addWidget(instructions)

        self.train_button = QPushButton("Train Model")
        self.train_button.setFixedHeight(50)
        self.train_button.clicked.connect(self.train_model)
        self.layout.addWidget(self.train_button)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 14px; margin-top: 20px; color: green;")
        self.layout.addWidget(self.status_label)

    def train_model(self):
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            recognizer = cv2.face.LBPHFaceRecognizer_create()

            image_paths, labels, label_names = self.prepare_training_data(face_cascade)

            if not image_paths:
                self.status_label.setText("No images found for training.")
                return

            recognizer.train(image_paths, np.array(labels))

            model_path = os.path.join("data", "models", "trained_model.yml")
            os.makedirs("data", "models", exist_ok=True)
            recognizer.save(model_path)

            label_path = os.path.join("data", "models", "label_names.pkl")
            with open(label_path, "wb") as label_file:
                pickle.dump(label_names, label_file)

            self.status_label.setText("Training completed successfully. Model saved!")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            self.status_label.setText("Error during training. Check logs for details.")

    def prepare_training_data(self, face_cascade):
        image_paths = []
        labels = []
        label_names = {}

        if not os.path.exists("data", "Training_images"):
            logging.error("Training image path does not exist.")
            return image_paths, labels, label_names

        for label_id, user_name in enumerate(os.listdir("data", "Training_images")):
            user_folder = os.path.join("data", "Training_images", user_name)
            if not os.path.isdir(user_folder):
                continue

            label_names[label_id] = user_name
            for file in os.listdir(user_folder):
                if file.endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(user_folder, file)

                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces:
                        face = image[y:y + h, x:x + w]
                        face = cv2.resize(face, (200, 200))
                        image_paths.append(face)
                        labels.append(label_id)
                        break

        return image_paths, labels, label_names