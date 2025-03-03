import cv2
from PyQt5.QtCore import * 
from PyQt5.QtGui import * 

class VideoCaptureThread(QThread):
    update_frame_signal = pyqtSignal(QPixmap)

    def __init__(self, camera_index, video_label=None):
        super().__init__()
        self.camera_index = camera_index
        self.video_label = video_label
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Convert frame to QPixmap
                height, width, channels = frame.shape
                bytes_per_line = channels * width
                qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(qt_image)
                self.update_frame_signal.emit(pixmap)
            else:
                break
        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()