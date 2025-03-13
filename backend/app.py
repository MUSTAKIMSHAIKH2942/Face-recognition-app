from flask import Flask, request, jsonify, send_from_directory
import os
from face_recognition import FaceRecognition
from video_stream import VideoStream
from utils.file_utils import ensure_folder_exists, load_json, save_json, append_to_csv
import logging
import cv2
import csv

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Paths
TRAIN_IMAGE_PATH = os.path.join("data", "Training_images")
UNKNOWN_IMAGE_PATH = os.path.join("data", "Unknown_images")
ATTENDANCE_PATH = os.path.join("data", "attendance.csv")
USERS_PATH = os.path.join("data", "users.csv")
MODEL_PATH = os.path.join("models", "face_recognition.yml")
LABEL_PATH = os.path.join("models", "label_names.pkl")

# Ensure folders exist
ensure_folder_exists(TRAIN_IMAGE_PATH)
ensure_folder_exists(UNKNOWN_IMAGE_PATH)
ensure_folder_exists("models")

# Initialize FaceRecognition
face_recognition = FaceRecognition(MODEL_PATH, LABEL_PATH, ATTENDANCE_PATH)

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/add_user", methods=["POST"])
def add_user():
    user_name = request.form.get("user_name")
    if not user_name:
        return jsonify({"error": "User name is required"}), 400

    # Create folder for user images
    user_folder = os.path.join(TRAIN_IMAGE_PATH, user_name)
    os.makedirs(user_folder, exist_ok=True)

    # Save user to CSV
    append_to_csv(USERS_PATH, [user_name, "Camera 1"])
    return jsonify({"message": f"User {user_name} added successfully"})

@app.route("/recognize", methods=["POST"])
def recognize():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UNKNOWN_IMAGE_PATH, file.filename)
    file.save(file_path)

    # Recognize faces in the image
    frame = cv2.imread(file_path)
    gray, faces = face_recognition.detect_faces(frame)
    recognized_faces = face_recognition.recognize_faces(gray, faces, frame)

    return jsonify({"recognized_faces": recognized_faces})

@app.route("/attendance", methods=["GET"])
def get_attendance():
    if not os.path.exists(ATTENDANCE_PATH):
        return jsonify({"error": "Attendance file not found"}), 404

    with open(ATTENDANCE_PATH, "r") as file:
        rows = list(csv.reader(file))
    return jsonify({"attendance": rows})

if __name__ == "__main__":
    app.run(debug=True)