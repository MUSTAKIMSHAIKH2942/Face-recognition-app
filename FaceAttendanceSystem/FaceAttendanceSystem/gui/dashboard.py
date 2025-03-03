from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QGridLayout, QScrollArea, QTextEdit
from PyQt5.QtCore import Qt
from face_attendance.video_stream import VideoCaptureThread
from ...FaceAttendanceSystem.gui.add_person import AddKnownPersonScreen
from ...FaceAttendanceSystem.gui.display_users import DisplayUsersScreen
import logging

class DashboardApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Attendance System")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.cameras = []
        self.video_widgets = {}
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
        self.video_grid_layout_with_buttons = QVBoxLayout(self.video_grid_widget)
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
        self.main_layout.addWidget(self.scroll_area, stretch=4)
        self.main_layout.addWidget(self.right_panel)

        # Draw initial grid
        self.draw_video_grid()

    def add_buttons(self):
        """Add buttons to the left panel."""
        self.add_known_person_button = QPushButton("Add Known Person", self)
        self.users_button = QPushButton("Users", self)
        self.license_button = QPushButton("License Validation", self)
        self.exit_button = QPushButton("Exit", self)

        self.left_panel_layout.addWidget(self.add_known_person_button)
        self.left_panel_layout.addWidget(self.users_button)
        self.left_panel_layout.addWidget(self.license_button)
        self.left_panel_layout.addWidget(self.exit_button)

        # Connect buttons to their respective functionalities
        self.add_known_person_button.clicked.connect(self.add_known_person)
        self.users_button.clicked.connect(self.show_users)
        self.license_button.clicked.connect(self.validate_license)
        self.exit_button.clicked.connect(self.exit_app)

    def add_known_person(self):
        """Open the Add Known Person screen."""
        self.add_known_person_window = AddKnownPersonScreen(self.cameras)
        self.add_known_person_window.show()

    def show_users(self):
        """Open the Display Users screen."""
        self.display_users_window = DisplayUsersScreen(self)
        self.display_users_window.show()

    def validate_license(self):
        """Validate the license."""
        QMessageBox.information(self, "License", "License validation logic goes here.")

    def exit_app(self):
        """Exit the application."""
        self.stop_all_streams()
        QApplication.quit()

    def stop_all_streams(self):
        """Stop all camera streams."""
        for thread in self.camera_threads.values():
            thread.stop()

    def draw_video_grid(self):
        """Draw the video grid dynamically based on the number of cameras."""
        for i in reversed(range(self.video_grid_layout.count())):
            widget = self.video_grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        self.grid_size = max(2, int(len(self.cameras) ** 0.5))
        self.video_grid_layout.setSpacing(5)
        self.video_grid_layout.setContentsMargins(5, 5, 5, 5)

        for index, camera_index in enumerate(self.cameras):
            row, col = divmod(index, self.grid_size)
            video_widget = self.create_video_widget(camera_index)
            self.video_grid_layout.addWidget(video_widget, row, col)

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

        video_label = QLabel(self)
        video_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_label.setMaximumSize(800, 450)
        video_label.setAlignment(Qt.AlignCenter)

        camera_thread = VideoCaptureThread(camera_index, video_label)
        camera_thread.update_frame_signal.connect(video_label.setPixmap)
        camera_thread.start()

        self.camera_threads[camera_index] = camera_thread

        delete_button = QPushButton("Delete", self)
        delete_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        delete_button.setFixedHeight(30)
        delete_button.clicked.connect(lambda: self.delete_camera(camera_index))

        video_layout.addWidget(video_label)
        video_layout.addWidget(delete_button)

        return video_widget

    def delete_camera(self, camera_index):
        if camera_index in self.camera_threads:
            thread = self.camera_threads.pop(camera_index)
            thread.stop()
            thread.wait()

        if camera_index in self.cameras:
            self.cameras.remove(camera_index)

        self.draw_video_grid()

    def add_camera(self):
        new_camera_index = len(self.cameras)
        self.cameras.append(new_camera_index)
        self.draw_video_grid()

    def log_message(self, message):
        """Log a message to the log area."""
        self.log_area.append(message)
        self.log_area.ensureCursorVisible()