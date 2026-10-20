import cv2

class CameraInterface:

    def __init__(self, source='droidcam', ip=None, port=4747):
        self.source = source
        self.ip = ip
        self.port = port
        self.cap = None

    def connect(self):

        if self.source == 'droidcam':

            url = f"http://{self.ip}:{self.port}/video"

            print("Connecting to:", url)

            self.cap = cv2.VideoCapture(url)

        else:
            self.cap = cv2.VideoCapture(0)

        return self.cap.isOpened()

    def read_frame(self):

        if self.cap is None:
            return None

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):

        if self.cap:
            self.cap.release()