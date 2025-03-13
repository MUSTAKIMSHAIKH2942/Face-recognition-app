import json
import pickle
import tkinter as tk
from tkinter import *
import os, cv2
import shutil
import csv
import numpy as np
from PIL import ImageTk, Image
import pandas as pd
import datetime
import time
import tkinter.ttk as tkk
import tkinter.font as font
import sys
from PyQt5.QtCore import QThread, pyqtSignal
import logging
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from datetime import datetime
from PyQt5.QtGui import *
from threading import Thread


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Paths for face recognition and attendance
haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = "TrainingImageLabel\\Trainner.yml"
trainimage_path = "Training_images"
userdetails_path = "UserDetails\\users.csv"
attendance_path = "Attendance"



class FaceRecognition:
    def __init__(self, camera_position="Camera 1", log_callback=None):
        # Load the trained model and label names
        self.model_path = os.path.join("models", "trained_model.yml")
        self.label_path = os.path.join("models", "label_names.pkl")
        self.attendance_file = "attendance.csv"
        self.camera_position = camera_position

        if not os.path.exists(self.model_path) or not os.path.exists(self.label_path):
            raise FileNotFoundError("Trained model or label file not found. Train the model first.")

        # Initialize the recognizer and load the model
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(self.model_path)

        # Load label names
        with open(self.label_path, "rb") as label_file:
            self.label_names = pickle.load(label_file)  # Dictionary: {label_id: user_name}

        # Initialize Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Create or validate the attendance file
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Date", "Name", "Camera Position", "In Time", "Out Time"])

        # Callback for logging messages
        self.log_callback = log_callback
        self.detected_persons = set()  # Track detected persons for the session

    def log_message(self, message):
        """Log a message using the callback if provided."""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def detect_faces(self, frame):
        """Detect faces in the frame using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return gray, faces

    def recognize_faces(self, gray_frame, faces):
        """Recognize faces and map to user names."""
        recognized_faces = []

        for (x, y, w, h) in faces:
            face = gray_frame[y:y + h, x:x + w]  # Crop detected face
            face_resized = cv2.resize(face, (200, 200))  # Resize to match training size

            # Predict the label using the recognizer
            label_id, confidence = self.recognizer.predict(face_resized)
            person_name = self.label_names.get(label_id, "Unknown")

            if person_name != "Unknown" and confidence < 50.0:  # Threshold for confident recognition
                # Record attendance and print log message for first-time detection
                self.record_attendance(person_name)
                if person_name not in self.detected_persons:
                    self.detected_persons.add(person_name)
                    self.log_message(f"Detected for the first time: {person_name}")

            recognized_faces.append((x, y, w, h, person_name, confidence))
        return recognized_faces

    
    def record_attendance(self, name):
        """Record attendance in the CSV file."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Check if the user already has an entry for today
        with open(self.attendance_file, "r") as file:
            rows = list(csv.reader(file))

        # Filter rows for today's date and the same user
        today_entries = [
            row for row in rows
            if row[0] == date_str and row[1] == name and row[2] == self.camera_position
        ]

        if not today_entries:
            # New entry for the user
            with open(self.attendance_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([date_str, name, self.camera_position, time_str, ""])
        else:
            # Update "Out Time" for the existing entry
            for i, row in enumerate(rows):
                if row[0] == date_str and row[1] == name and row[2] == self.camera_position:
                    rows[i][4] = time_str  # Update "Out Time"
                    break
            with open(self.attendance_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(rows)

    def update_frame_with_recognition(self, frame):
        """Detect, recognize faces, and draw rectangles with labels on the frame."""
        gray_frame, faces = self.detect_faces(frame)
        recognized_faces = self.recognize_faces(gray_frame, faces)

        for (x, y, w, h, person_name, confidence) in recognized_faces:
            color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{person_name} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
        return frame

        
# Video Stream and Capture Threading for GUI
class VideoStream:
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)
        self.running = True
        self.face_recognition = FaceRecognition()

    def start_stream(self):
        if not self.capture.isOpened():
            raise Exception(f"Camera {self.camera_index} not available")

    def get_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = self.face_recognition.update_frame_with_recognition(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(img)
        return None

    def stop_stream(self):
        self.capture.release()

class VideoCaptureThread(QThread):
    update_frame_signal = pyqtSignal(QPixmap)

    def __init__(self, camera_index, video_label=None):
        super().__init__()
        self.camera_index = camera_index
        self.video_label = video_label  # Store video label if needed
        self.running = True
        self.face_recognition = FaceRecognition()

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        while not self.isInterruptionRequested():
            ret, frame = cap.read()
            if ret:
                # Apply face detection and recognition
                frame = self.face_recognition.update_frame_with_recognition(frame)

                # Convert frame to QPixmap for display
                height, width, channels = frame.shape
                bytes_per_line = channels * width
                qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(qt_image)

                # Emit updated frame
                self.update_frame_signal.emit(pixmap.scaled(320, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                break
        cap.release()

    def stop(self):
        """Stop the video capture thread."""
        self.running = False
        self.quit()  # Signal the thread to exit
        self.wait()  # Ensure the thread finishes cleanly

    def convert_frame_to_qimage(self, frame):
        """Convert a frame (NumPy array) to QImage."""
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        return QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)





class AddKnownPersonScreen(QWidget):
    def __init__(self, cameras, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Known Person")
        self.setGeometry(150, 150, 800, 600)
        self.cameras = cameras
        self.layout = QVBoxLayout(self)
        self.setup_ui()

    def setup_ui(self):
        # Instructions
        instructions = QLabel("Choose an option to add a known person:")
        instructions.setStyleSheet("font-size: 16px; margin-bottom: 20px;")
        self.layout.addWidget(instructions)

        # Buttons for options
        self.capture_button = QPushButton("Capture Images from Camera")
        self.upload_button = QPushButton("Upload Images")
        self.train_button = QPushButton("Train Model (after adding images)")

        self.capture_button.setFixedHeight(50)
        self.upload_button.setFixedHeight(50)
        self.train_button.setFixedHeight(50)

        self.capture_button.clicked.connect(self.capture_images)
        self.upload_button.clicked.connect(self.upload_images)
        self.train_button.clicked.connect(self.train_model)

        self.layout.addWidget(self.capture_button)
        self.layout.addWidget(self.upload_button)
        self.layout.addWidget(self.train_button)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 14px; margin-top: 20px; color: green;")
        self.layout.addWidget(self.status_label)

    def capture_images(self):
        self.capture_window = CaptureImagesScreen(self.cameras)
        self.capture_window.show()
        self.close()

    def upload_images(self):
        self.upload_window = UploadImagesScreen()
        self.upload_window.show()
        self.close()

    
    def train_model(self):
        """Train the model with added images."""
        try:
            # Initialize face detector and recognizer
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            recognizer = cv2.face.LBPHFaceRecognizer_create()

            # Prepare training data
            image_paths, labels, label_names = self.prepare_training_data(face_cascade)

            if not image_paths:
                self.status_label.setText("No images found for training.")
                return

            # Train the recognizer
            recognizer.train(image_paths, np.array(labels))

            # Save the trained model and label names
            model_path = os.path.join("models", "trained_model.yml")
            os.makedirs("models", exist_ok=True)
            recognizer.save(model_path)

            label_path = os.path.join("models", "label_names.pkl")
            with open(label_path, "wb") as label_file:
                pickle.dump(label_names, label_file)

            self.status_label.setText("Training completed successfully. Model saved!")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            self.status_label.setText("Error during training. Check logs for details.")

    def prepare_training_data(self, face_cascade):
        """Prepares images and labels for training."""
        image_paths = []
        labels = []
        label_names = {}

        if not os.path.exists(trainimage_path):
            logging.error("Training image path does not exist.")
            return image_paths, labels, label_names

        for label_id, user_name in enumerate(os.listdir(trainimage_path)):
            user_folder = os.path.join(trainimage_path, user_name)
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

class CaptureImagesScreen(QWidget):
    def __init__(self, cameras, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture Images")
        self.setGeometry(200, 200, 1000, 700)
        self.cameras = cameras
        self.capture_flag = False
        self.frame_count = 0
        self.setup_ui()

    def setup_ui(self):
        # Main layout
        self.main_layout = QVBoxLayout(self)

        # Header layout (for camera selection and user input)
        self.header_layout = QHBoxLayout()
        self.main_layout.addLayout(self.header_layout)

        # Camera selection
        self.camera_label = QLabel("Select Camera:")
        self.camera_label.setStyleSheet("font-size: 14px;")
        self.camera_dropdown = QComboBox(self)
        self.camera_dropdown.addItems([f"Camera {i}" for i in range(len(self.cameras))])
        self.camera_dropdown.setCurrentIndex(0)
        self.header_layout.addWidget(self.camera_label)
        self.header_layout.addWidget(self.camera_dropdown)

        # User input
        self.user_label = QLabel("Enter User Name:")
        self.user_label.setStyleSheet("font-size: 14px;")
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter the name of the person...")
        self.header_layout.addWidget(self.user_label)
        self.header_layout.addWidget(self.user_input)

        # Video feed display
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("border: 1px solid black;")
        self.main_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Footer layout (for buttons)
        self.footer_layout = QHBoxLayout()
        self.main_layout.addLayout(self.footer_layout)

        # Buttons
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        self.footer_layout.addWidget(self.start_button)

        
        self.capture_button = QPushButton("Capture Frame")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.capture_frame)
        self.footer_layout.addWidget(self.capture_button)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_camera)
        self.footer_layout.addWidget(self.stop_button)

    def start_camera(self):
        self.selected_camera_index = self.camera_dropdown.currentIndex()
        self.user_name = self.user_input.text().strip()

        if not self.user_name:
            QMessageBox.warning(self, "Error", "User name cannot be empty!")
            return

        # Ensure folder exists
        self.user_folder = os.path.join(trainimage_path, self.user_name)
        os.makedirs(self.user_folder, exist_ok=True)

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.selected_camera_index)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open the selected camera.")
            return

        self.capture_flag = True
        self.capture_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)

        # Start updating the video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (800, 600))  # Resize frame to fit the video label
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap)

    def capture_frame(self):
        if self.capture_flag and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                file_path = os.path.join(self.user_folder, f"{self.user_name}_{self.frame_count}.jpg")
                cv2.imwrite(file_path, frame)
                self.frame_count += 1
                logging.info(f"Captured frame {self.frame_count} for {self.user_name}")

                # Stop after capturing 500 frames
                if self.frame_count >= 500:
                    QMessageBox.information(self, "Info", "Captured 500 frames. Stopping...")
                    self.stop_camera()

    def stop_camera(self):
        self.capture_flag = False
        self.capture_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()

        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

        self.video_label.clear()

        if self.frame_count > 0:
            # Check if the user already exists in the CSV file
            with open(userdetails_path, "r") as file:
                reader = csv.DictReader(file)
                existing_users = {row['user_name'] for row in reader}

            # Log new user if not already in CSV
            if self.user_name not in existing_users:
                with open(userdetails_path, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([self.user_name, self.selected_camera_index])
                logging.info(f"Saved {self.frame_count} frames for {self.user_name} in {self.user_folder}")
                QMessageBox.information(self, "Info", f"Captured {self.frame_count} frames for {self.user_name}.")
            else:
                logging.info(f"User {self.user_name} already exists in CSV.")

        self.frame_count = 0

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)



class UploadImagesScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Upload Images")
        self.setGeometry(200, 200, 900, 600)
        self.layout = QVBoxLayout(self)
        self.setup_ui()

    def setup_ui(self):
        # Enter user details
        self.user_label = QLabel("Enter User Name:")
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter the name of the person...")

        self.layout.addWidget(self.user_label)
        self.layout.addWidget(self.user_input)

        # Upload button
        self.upload_button = QPushButton("Upload Images")
        self.upload_button.clicked.connect(self.upload_images)
        self.layout.addWidget(self.upload_button)

    def upload_images(self):
        user_name = self.user_input.text().strip()

        if not user_name:
            logging.error("User name cannot be empty!")
            return

        # Select files
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg)")
        if not files:
            logging.error("No images selected!")
            return

        # Ensure folder exists
        user_folder = os.path.join(trainimage_path, user_name)
        os.makedirs(user_folder, exist_ok=True)

        # Copy files
        for file in files:
            shutil.copy(file, user_folder)

        logging.info(f"Uploaded images for {user_name} stored in {user_folder}")

        # Log to CSV
        with open(userdetails_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([user_name, "Uploaded"])

        self.close()

class TrainImagesScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train Images")
        self.setGeometry(200, 200, 800, 600)
        self.layout = QVBoxLayout(self)
        self.setup_ui()

    def setup_ui(self):
        # Title
        title = QLabel("Train Images for Face Recognition")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        self.layout.addWidget(title)

        # Instructions
        instructions = QLabel("Click the button below to start training the model with available images.")
        instructions.setStyleSheet("font-size: 14px; margin-bottom: 20px;")
        self.layout.addWidget(instructions)

        # Train button
        self.train_button = QPushButton("Train Model")
        self.train_button.setFixedHeight(50)
        self.train_button.clicked.connect(self.train_model)
        self.layout.addWidget(self.train_button)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 14px; margin-top: 20px; color: green;")
        self.layout.addWidget(self.status_label)

    



class DashboardApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Attendance System")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.cameras = []
        self.video_widgets = {}
        self.camera_indices = self.load_camera_indices()
        self.camera_threads = {}
        self.grid_size = 2

        # Setup layout and widgets
        self.main_layout = QHBoxLayout(self)

        # Left panel
        self.left_panel = QFrame(self)
        self.left_panel.setFixedWidth(180)
        self.left_panel.setStyleSheet("background-color: lightgray; padding: 10px;")
        self.left_panel_layout = QVBoxLayout(self.left_panel)
        self.add_buttons()

        # Right panel for logs
        self.right_panel = QFrame(self)
        self.right_panel.setFixedWidth(200)
        self.right_panel.setStyleSheet("background-color: #f0f0f0;")
        self.log_area = QTextEdit(self.right_panel)
        self.log_area.setReadOnly(True)
        self.right_panel_layout = QVBoxLayout(self.right_panel)
        self.right_panel_layout.addWidget(self.log_area)

        # Center panel for video streams and buttons
        self.video_grid_layout = QGridLayout()
        self.video_grid_layout.setSpacing(10)

        # Scroll area for video grid
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.video_grid_widget = QWidget()
        self.video_grid_layout_with_buttons = QVBoxLayout(self.video_grid_widget)  # Wrap grid in vertical layout
        self.video_grid_layout_with_buttons.addLayout(self.video_grid_layout)

        # Add Load and Save buttons
        self.button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Cameras", self)
        self.load_button.setFixedSize(120, 40)
        self.load_button.clicked.connect(self.load_and_update_cameras)
        self.save_button = QPushButton("Save Cameras", self)
        self.save_button.setFixedSize(120, 40)
        self.save_button.clicked.connect(self.save_camera_indices)
        self.add_ip_camera_button = QPushButton("Add IP Camera", self)
        self.add_ip_camera_button.setFixedSize(120, 40)
        self.add_ip_camera_button.clicked.connect(self.open_add_ip_camera_window)
        self.button_layout.addWidget(self.load_button)
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.add_ip_camera_button)

        self.video_grid_layout_with_buttons.addLayout(self.button_layout)

        


        # Add widget to scroll area
        self.scroll_area.setWidget(self.video_grid_widget)

        # Add panels to main layout
        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.scroll_area, stretch=4)  # Larger stretch factor for center
        self.main_layout.addWidget(self.right_panel)

        # Draw initial grid
        self.draw_video_grid()

    def add_buttons(self):
        """Add buttons to the left panel."""
        # Existing buttons
        # self.add_dashboard_button = QPushButton("Dashboard", self)
        self.add_known_person_button = QPushButton("Add Known Person", self)
        self.users_button = QPushButton("Users", self)
        self.license_button = QPushButton("License Validation", self)
        self.exit_button = QPushButton("Exit", self)


        # Add widgets to the left panel layout
        # self.left_panel_layout.addWidget(self.add_dashboard_button)
        self.left_panel_layout.addWidget(self.add_known_person_button)
        self.left_panel_layout.addWidget(self.users_button)
        self.left_panel_layout.addWidget(self.license_button)

        self.left_panel_layout.addWidget(self.exit_button)

        # Connect buttons to their respective functionalities
        # self.add_dashboard_button.clicked.connect(self.show_dashboard)
        self.add_known_person_button.clicked.connect(self.add_known_person)
        self.users_button.clicked.connect(self.show_users)
        self.license_button.clicked.connect(self.validate_license)
        self.exit_button.clicked.connect(self.exit_app)


    def open_add_ip_camera_window(self):
        """Open a window to add an IP-based camera."""
        dialog = AddIPCameraDialog(self)
        dialog.exec_()


    def draw_video_grid(self):
        """Draw the video grid dynamically based on the number of cameras."""
        # Clear existing widgets
        for i in reversed(range(self.video_grid_layout.count())):
            widget = self.video_grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Calculate rows and columns dynamically based on the number of cameras
        self.grid_size = max(2, int(len(self.cameras) ** 0.5))
        self.video_grid_layout.setSpacing(5)
        self.video_grid_layout.setContentsMargins(5, 5, 5, 5)

        # Draw video widgets for each camera
        for index, camera_index in enumerate(self.cameras):
            row, col = divmod(index, self.grid_size)
            video_widget = self.create_video_widget(camera_index)
            self.video_grid_layout.addWidget(video_widget, row, col)

        # Add a "+" button for new cameras
        if len(self.cameras) < self.grid_size ** 2:
            add_button = QPushButton("+", self)
            add_button.setFixedSize(60, 60)
            add_button.setStyleSheet("border-radius: 30px; background-color: lightgray; font-size: 18px;")
            add_button.clicked.connect(self.add_camera)
            row, col = divmod(len(self.cameras), self.grid_size)
            self.video_grid_layout.addWidget(add_button, row, col)





    def create_video_widget(self, camera_index):
        """Create a QLabel for video display and start the camera thread."""
        video_widget = QWidget(self)
        video_layout = QVBoxLayout(video_widget)

        # Video label
        video_label = QLabel(self)
        video_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_label.setMaximumSize(800, 450)
        video_label.setAlignment(Qt.AlignCenter)

        # Start video capture thread
        camera_thread = VideoCaptureThread(camera_index, video_label)
        camera_thread.update_frame_signal.connect(video_label.setPixmap)
        camera_thread.start()

        # Store thread reference to manage lifecycle
        self.camera_threads[camera_index] = camera_thread

        # Add delete button
        delete_button = QPushButton("Delete", self)
        delete_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        delete_button.setFixedHeight(30)
        delete_button.clicked.connect(lambda: self.delete_camera(camera_index))

        # Add video label and delete button to layout
        video_layout.addWidget(video_label)
        video_layout.addWidget(delete_button)

        return video_widget




    

    def update_frame(self, frame, label):
        """Update the video frame in the specified label."""
        if frame is not None:
            # Convert frame (NumPy array) to QImage
            height, width, channels = frame.shape
            bytes_per_line = channels * width
            qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
            
            # Scale the QImage to fit the label and set it
            scaled_frame = qt_image.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(QPixmap.fromImage(scaled_frame))


    def load_and_update_cameras(self):
        """Load camera indices from a file and update the grid."""
        self.camera_indices = self.load_camera_indices()  # Load from file
        self.cameras = self.camera_indices[:]  # Sync the loaded indices with the cameras list
        self.draw_video_grid()  # Redraw the grid
        self.log_message(f"Loaded cameras: {self.camera_indices}")  # Optional logging


    def add_camera(self):
        """Add a new camera."""
        new_camera_index = len(self.cameras)  # New index based on current count
        self.cameras.append(new_camera_index)  # Add to the list
        self.camera_indices.append(new_camera_index)  # Save to persistent storage
        self.save_camera_indices()  # Save updated indices
        self.draw_video_grid()  # Redraw grid
        self.log_message(f"Added new camera: {new_camera_index}")

    def delete_camera(self, camera_index):
        print(f"Deleting camera: {camera_index}")  # Debug print
        if camera_index in self.camera_threads:
            thread = self.camera_threads.pop(camera_index)
            thread.stop()
            thread.wait()

        if camera_index in self.cameras:
            self.cameras.remove(camera_index)

        if camera_index in self.camera_indices:
            self.camera_indices.remove(camera_index)
            self.save_camera_indices()

        self.draw_video_grid()
        self.log_message(f"Deleted camera: {camera_index}")




    def log_message(self, message):
        """Log a message to the log area."""
        self.log_area.append(message)
        self.log_area.ensureCursorVisible()



    def stop_all_streams(self):
        """Stop all camera streams."""
        for thread in self.camera_threads.values():
            thread.stop()

    # def show_dashboard(self):
    #     logging.info("Showing dashboard")

    def add_known_person(self):
        self.add_known_person_window = AddKnownPersonScreen(self.camera_indices)
        self.add_known_person_window.show()

    def show_users(self):
        """Open the Display Users window."""
        self.display_users_window = DisplayUsersScreen(self)
        self.display_users_window.show()

    def validate_license(self):
        """Validate the license based on the number of unique users and display users in the frontend."""
        try:
            # Track unique users based on user_name
            unique_users = set()  # Set to store unique user names

            with open(userdetails_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)

                # Print the fieldnames (column headers)
                logging.info(f"CSV Headers: {reader.fieldnames}")  # Log the headers

                for row in reader:
                    # Check if 'user_name' exists in the current row
                    if 'user_name' in row:
                        unique_users.add(row['user_name'])
                    else:
                        logging.warning(f"No 'user_name' in row: {row}")  # Log missing user_name

            # Get the unique user count
            user_count = len(unique_users)

            # Check if the user limit is exceeded
            if user_count > 20:
                logging.info(f"User count ({user_count}) exceeded the limit of 20. Prompting for license key.")
                self.show_license_dialog(unique_users)  # Pass the unique_users set to the dialog
            else:
                # Show users in a QMessageBox if the count is under 20
                user_list = "\n".join(unique_users)  # Convert the set to a string with line breaks
                QMessageBox.information(
                    self,
                    "License Valid",
                    f"License is valid. Users:\n{user_list}"
                )
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Users CSV file not found.")
            logging.error("Users CSV file not found.")


    def show_license_dialog(self, unique_users):
        """Display a dialog for entering the license key and show the list of users."""
        dialog = QDialog(self)
        dialog.setWindowTitle("License Validation")
        dialog.setFixedSize(400, 300)  # Increased size to accommodate user list

        layout = QVBoxLayout(dialog)

        # Instruction label
        label = QLabel("You have exceeded the user limit. Please enter a valid license key:")
        layout.addWidget(label)

        # License key input
        license_input = QLineEdit(dialog)
        license_input.setPlaceholderText("Enter your license key")
        layout.addWidget(license_input)

        # Display the list of users
        user_list_label = QLabel("Current Users:")
        layout.addWidget(user_list_label)

        # Create a scrollable area for the user list
        scroll_area = QScrollArea(dialog)
        scroll_area.setWidgetResizable(True)
        user_list_widget = QWidget()
        user_list_layout = QVBoxLayout(user_list_widget)

        # Add each user to the list
        for user in unique_users:
            user_label = QLabel(user)
            user_list_layout.addWidget(user_label)

        scroll_area.setWidget(user_list_widget)
        layout.addWidget(scroll_area)

        # Submit button
        submit_button = QPushButton("Validate", dialog)
        layout.addWidget(submit_button)

        def validate_key():
            static_license_key = "MY-VALID-KEY"  # Static license key
            if license_input.text() == static_license_key:
                QMessageBox.information(dialog, "Success", "License validated successfully!")
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Invalid Key", "The license key you entered is invalid.")

        submit_button.clicked.connect(validate_key)
        dialog.exec_()
    def exit_app(self):
        self.stop_all_streams()
        self.save_camera_indices()
        QApplication.quit()

    def save_camera_indices(self):
        """Save camera indices to a file."""
        with open("camera_indices.json", "w") as file:
            json.dump(self.camera_indices, file)
        self.log_message("Camera indices saved: " + str(self.camera_indices))

    def load_camera_indices(self):
        """Load camera indices from a file."""
        try:
            with open("camera_indices.json", "r") as file:
                indices = json.load(file)
                # self.log_message("Camera indices loaded: " + str(indices))
                return indices
        except FileNotFoundError:
            self.log_message("No saved camera indices found.")
            return []

class AddIPCameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add IP Camera")
        self.setGeometry(200, 200, 400, 200)

        # Layout
        self.layout = QVBoxLayout(self)

        # Input fields
        self.ip_label = QLabel("IP Address:")
        self.ip_input = QLineEdit(self)
        self.name_label = QLabel("Camera Name:")
        self.name_input = QLineEdit(self)

        self.layout.addWidget(self.ip_label)
        self.layout.addWidget(self.ip_input)
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.name_input)

        # Buttons
        self.button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add", self)
        self.add_button.clicked.connect(self.add_camera)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.close)

        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.cancel_button)

        self.layout.addLayout(self.button_layout)

    def add_camera(self):
        """Add the IP camera details."""
        ip_address = self.ip_input.text().strip()
        camera_name = self.name_input.text().strip()

        if not ip_address or not camera_name:
            self.parent.log_area.append("Both fields are required.")
            return

        # Save to camera indices
        self.parent().camera_indices.append({"name": camera_name, "ip": ip_address})
        self.parent().log_area.append(f"Added IP Camera: {camera_name} ({ip_address})")
        self.close()


class DisplayUsersScreen(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Users in the System")
        self.setGeometry(200, 200, 900, 600)
        self.layout = QVBoxLayout(self)
        self.setup_ui()

    def setup_ui(self):
        # Title for the dialog
        title = QLabel("Users in the System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 20px;")
        self.layout.addWidget(title)

        # Load user data
        self.users_data = self.load_user_data()

        # If no users found, show a message
        if not self.users_data:
            no_users_label = QLabel("No users found.")
            no_users_label.setAlignment(Qt.AlignCenter)
            no_users_label.setStyleSheet("font-size: 14px; color: red; margin-top: 20px;")
            self.layout.addWidget(no_users_label)
            return

        # Create a scrollable area for user list
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area_content = QWidget(scroll_area)
        scroll_area.setWidget(scroll_area_content)
        scroll_area_layout = QVBoxLayout(scroll_area_content)
        scroll_area_layout.setSpacing(15)

        # Display each user in the system
        for user_name in self.users_data:
            user_widget = self.create_user_widget(user_name)
            scroll_area_layout.addWidget(user_widget)

        self.layout.addWidget(scroll_area)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        close_button.setFixedHeight(40)
        close_button.setStyleSheet("margin-top: 20px; background-color: #f05454; color: white; font-size: 16px;")
        self.layout.addWidget(close_button)

    def create_user_widget(self, user_name):
        """Creates a widget for a single user."""
        user_widget = QWidget()
        user_widget_layout = QVBoxLayout(user_widget)

        # User image
        user_image_path = os.path.join(trainimage_path, user_name, f"{user_name}_0.jpg")
        user_image_label = QLabel(self)
        if os.path.exists(user_image_path):
            user_image = QPixmap(user_image_path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            user_image = self.make_image_circle(user_image)  # Make the image circular
            user_image_label.setPixmap(user_image)
        else:
            user_image_label.setText("No Image")
            user_image_label.setAlignment(Qt.AlignCenter)
            user_image_label.setStyleSheet("font-size: 12px; color: red; background-color: #d3d3d3; border-radius: 50px; height: 100px; width: 100px;")

        # User name
        user_label = QLabel(user_name)
        user_label.setAlignment(Qt.AlignCenter)
        user_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")

        # Delete button
        delete_button = QPushButton("Delete")
        delete_button.setFixedHeight(30)
        delete_button.setStyleSheet("background-color: #ff6b6b; color: white; font-size: 14px;")
        delete_button.clicked.connect(lambda: self.delete_user(user_name))

        # Layout adjustments
        user_widget_layout.addWidget(user_image_label, alignment=Qt.AlignCenter)
        user_widget_layout.addWidget(user_label, alignment=Qt.AlignCenter)
        user_widget_layout.addWidget(delete_button, alignment=Qt.AlignCenter)

        return user_widget

    def make_image_circle(self, pixmap):
        """Converts a QPixmap to a circular shape."""
        size = min(pixmap.width(), pixmap.height())
        circular_pixmap = QPixmap(size, size)
        circular_pixmap.fill(Qt.transparent)

        painter = QPainter(circular_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addEllipse(0, 0, size, size)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        return circular_pixmap

    def load_user_data(self):
        """Loads user data from the CSV file."""
        try:
            if not os.path.exists(userdetails_path):
                return []

            with open(userdetails_path, "r") as file:
                users = [line.strip().split(",")[0] for line in file.readlines()]
            return users
        except Exception as e:
            logging.error(f"Error loading user data: {e}")
            return []

    def delete_user(self, user_name):
        """Deletes a user from the CSV file and updates the UI."""
        try:
            # Load current user data
            with open(userdetails_path, "r") as file:
                users = file.readlines()

            # Remove the user entry
            with open(userdetails_path, "w") as file:
                for line in users:
                    if not line.strip().startswith(user_name):
                        file.write(line)

            # Remove user's images folder
            user_folder = os.path.join(trainimage_path, user_name)
            if os.path.exists(user_folder):
                import shutil
                shutil.rmtree(user_folder)

            # Update the UI
            self.users_data.remove(user_name)
            self.refresh_ui()

        except Exception as e:
            logging.error(f"Error deleting user {user_name}: {e}")

    def refresh_ui(self):
        """Refreshes the UI after deleting a user."""
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.setup_ui()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DashboardApp()
    window.show()
    sys.exit(app.exec_())
