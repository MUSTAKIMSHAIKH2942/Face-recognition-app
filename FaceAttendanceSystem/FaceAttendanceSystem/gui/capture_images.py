from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QHBoxLayout, QComboBox, QLineEdit
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import os
import logging
import csv

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
        self.main_layout = QVBoxLayout(self)

        self.header_layout = QHBoxLayout()
        self.main_layout.addLayout(self.header_layout)

        self.camera_label = QLabel("Select Camera:")
        self.camera_label.setStyleSheet("font-size: 14px;")
        self.camera_dropdown = QComboBox(self)
        self.camera_dropdown.addItems([f"Camera {i}" for i in range(len(self.cameras))])
        self.camera_dropdown.setCurrentIndex(0)
        self.header_layout.addWidget(self.camera_label)
        self.header_layout.addWidget(self.camera_dropdown)

        self.user_label = QLabel("Enter User Name:")
        self.user_label.setStyleSheet("font-size: 14px;")
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter the name of the person...")
        self.header_layout.addWidget(self.user_label)
        self.header_layout.addWidget(self.user_input)

        self.video_label = QLabel(self)
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("border: 1px solid black;")
        self.main_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        self.footer_layout = QHBoxLayout()
        self.main_layout.addLayout(self.footer_layout)

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

        self.user_folder = os.path.join("data", "Training_images", self.user_name)
        os.makedirs(self.user_folder, exist_ok=True)

        self.cap = cv2.VideoCapture(self.selected_camera_index)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open the selected camera.")
            return

        self.capture_flag = True
        self.capture_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (800, 600))
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
            with open("data", "users.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.user_name, self.selected_camera_index])
            logging.info(f"Saved {self.frame_count} frames for {self.user_name} in {self.user_folder}")
            QMessageBox.information(self, "Info", f"Captured {self.frame_count} frames for {self.user_name}.")

        self.frame_count = 0

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)