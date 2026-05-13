import cv2
import threading
import queue

class VideoStream:
    def __init__(self, source, queue_size=128):
        self.source = source
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise ValueError(f"Unable to open video source: {source}")
            
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            if not self.frame_queue.full():
                ret, frame = self.capture.read()
                if not ret:
                    self.stop()
                    break
                self.frame_queue.put(frame)
            else:
                # Slight delay if queue is full to prevent high CPU usage
                cv2.waitKey(1)
                
    def read(self):
        if not self.frame_queue.empty():
            return True, self.frame_queue.get()
        return False, None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.capture.release()
