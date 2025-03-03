from PyQt5.QtWidgets import * 
import os
import shutil
import logging
import csv

class UploadImagesScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Upload Images")
        self.setGeometry(200, 200, 900, 600)
        self.layout = QVBoxLayout(self)
        self.setup_ui()

    def setup_ui(self):
        self.user_label = QLabel("Enter User Name:")
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter the name of the person...")

        self.layout.addWidget(self.user_label)
        self.layout.addWidget(self.user_input)

        self.upload_button = QPushButton("Upload Images")
        self.upload_button.clicked.connect(self.upload_images)
        self.layout.addWidget(self.upload_button)

    def upload_images(self):
        user_name = self.user_input.text().strip()

        if not user_name:
            logging.error("User name cannot be empty!")
            return

        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg)")
        if not files:
            logging.error("No images selected!")
            return

        user_folder = os.path.join("data", "Training_images", user_name)
        os.makedirs(user_folder, exist_ok=True)

        for file in files:
            shutil.copy(file, user_folder)

        logging.info(f"Uploaded images for {user_name} stored in {user_folder}")

        with open("data", "users.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([user_name, "Uploaded"])

        self.close()