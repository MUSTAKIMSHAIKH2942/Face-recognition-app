import cv2
from threading import Thread

class VideoStream:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)
        self.running = False

    def start(self):
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                break

    def read(self):
        return self.capture.read()

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()