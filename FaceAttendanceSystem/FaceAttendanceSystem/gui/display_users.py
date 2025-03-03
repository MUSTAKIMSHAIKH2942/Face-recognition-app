from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QScrollArea, QWidget, QMessageBox
from PyQt5.QtGui import QPixmap, QPainter, QPainterPath
from PyQt5.QtCore import Qt
import os
import csv
import logging

class DisplayUsersScreen(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Users in the System")
        self.setGeometry(200, 200, 900, 600)
        self.layout = QVBoxLayout(self)
        self.setup_ui()

    def setup_ui(self):
        title = QLabel("Users in the System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 20px;")
        self.layout.addWidget(title)

        self.users_data = self.load_user_data()

        if not self.users_data:
            no_users_label = QLabel("No users found.")
            no_users_label.setAlignment(Qt.AlignCenter)
            no_users_label.setStyleSheet("font-size: 14px; color: red; margin-top: 20px;")
            self.layout.addWidget(no_users_label)
            return

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area_content = QWidget(scroll_area)
        scroll_area.setWidget(scroll_area_content)
        scroll_area_layout = QVBoxLayout(scroll_area_content)
        scroll_area_layout.setSpacing(15)

        for user_name in self.users_data:
            user_widget = self.create_user_widget(user_name)
            scroll_area_layout.addWidget(user_widget)

        self.layout.addWidget(scroll_area)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        close_button.setFixedHeight(40)
        close_button.setStyleSheet("margin-top: 20px; background-color: #f05454; color: white; font-size: 16px;")
        self.layout.addWidget(close_button)

    def create_user_widget(self, user_name):
        user_widget = QWidget()
        user_widget_layout = QVBoxLayout(user_widget)

        user_image_path = os.path.join("data", "Training_images", user_name, f"{user_name}_0.jpg")
        user_image_label = QLabel(self)
        if os.path.exists(user_image_path):
            user_image = QPixmap(user_image_path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            user_image = self.make_image_circle(user_image)
            user_image_label.setPixmap(user_image)
        else:
            user_image_label.setText("No Image")
            user_image_label.setAlignment(Qt.AlignCenter)
            user_image_label.setStyleSheet("font-size: 12px; color: red; background-color: #d3d3d3; border-radius: 50px; height: 100px; width: 100px;")

        user_label = QLabel(user_name)
        user_label.setAlignment(Qt.AlignCenter)
        user_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")

        delete_button = QPushButton("Delete")
        delete_button.setFixedHeight(30)
        delete_button.setStyleSheet("background-color: #ff6b6b; color: white; font-size: 14px;")
        delete_button.clicked.connect(lambda: self.delete_user(user_name))

        user_widget_layout.addWidget(user_image_label, alignment=Qt.AlignCenter)
        user_widget_layout.addWidget(user_label, alignment=Qt.AlignCenter)
        user_widget_layout.addWidget(delete_button, alignment=Qt.AlignCenter)

        return user_widget

    def make_image_circle(self, pixmap):
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
        try:
            if not os.path.exists("data", "users.csv"):
                return []

            with open("data", "users.csv", "r") as file:
                users = [line.strip().split(",")[0] for line in file.readlines()]
            return users
        except Exception as e:
            logging.error(f"Error loading user data: {e}")
            return []

    def delete_user(self, user_name):
        try:
            with open("data", "users.csv", "r") as file:
                users = file.readlines()

            with open("data", "users.csv", "w") as file:
                for line in users:
                    if not line.strip().startswith(user_name):
                        file.write(line)

            user_folder = os.path.join("data", "Training_images", user_name)
            if os.path.exists(user_folder):
                import shutil
                shutil.rmtree(user_folder)

            self.users_data.remove(user_name)
            self.refresh_ui()

        except Exception as e:
            logging.error(f"Error deleting user {user_name}: {e}")

    def refresh_ui(self):
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.setup_ui()
