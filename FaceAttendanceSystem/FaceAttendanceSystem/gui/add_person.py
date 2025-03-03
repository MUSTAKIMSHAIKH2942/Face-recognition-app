from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from ...FaceAttendanceSystem.gui.capture_images import CaptureImagesScreen
from ...FaceAttendanceSystem.gui.upload_images import UploadImagesScreen

class AddKnownPersonScreen(QWidget):
    def __init__(self, cameras, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Known Person")
        self.setGeometry(150, 150, 800, 600)
        self.cameras = cameras
        self.layout = QVBoxLayout(self)
        self.setup_ui()

    def setup_ui(self):
        instructions = QLabel("Choose an option to add a known person:")
        instructions.setStyleSheet("font-size: 16px; margin-bottom: 20px;")
        self.layout.addWidget(instructions)

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
        self.status_label.setText("Training model...")
        QMessageBox.information(self, "Training", "Model training logic goes here.")