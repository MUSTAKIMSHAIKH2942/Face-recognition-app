import cv2
import os
import numpy as np
from face_recognition import FaceRecognition

# Paths
TRAIN_IMAGE_PATH = os.path.join("data", "Training_images")
MODEL_PATH = os.path.join("models", "face_recognition.yml")
LABEL_PATH = os.path.join("models", "label_names.pkl")

# Ensure folders exist
os.makedirs("models", exist_ok=True)

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Prepare training data
def prepare_training_data():
    image_paths = []
    labels = []
    label_names = {}

    for label_id, user_name in enumerate(os.listdir(TRAIN_IMAGE_PATH)):
        user_folder = os.path.join(TRAIN_IMAGE_PATH, user_name)
        if not os.path.isdir(user_folder):
            continue

        label_names[label_id] = user_name  # Map label IDs to user names
        for file in os.listdir(user_folder):
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(user_folder, file)

                # Read image and detect face
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # Use the first detected face for training
                for (x, y, w, h) in faces:
                    face = image[y:y + h, x:x + w]
                    face = cv2.resize(face, (200, 200))  # Resize to uniform size
                    image_paths.append(face)
                    labels.append(label_id)
                    break  # Consider only the first face in the image

    return image_paths, labels, label_names

# Train the model
def train_model():
    # Prepare training data
    image_paths, labels, label_names = prepare_training_data()

    if not image_paths:
        print("No training data found. Add images to the Training_images folder.")
        return

    # Initialize the recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the recognizer
    recognizer.train(image_paths, np.array(labels))

    # Save the trained model and label names
    recognizer.save(MODEL_PATH)
    with open(LABEL_PATH, "wb") as f:
        import pickle
        pickle.dump(label_names, f)

    print(f"Model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()