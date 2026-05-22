import cv2

class VideoWriter:
    def __init__(self, output_path, fps, width, height):
        self.output_path = output_path
        
        # We use the mp4v codec for standard local MP4 files. 
        # For RTMP streams, FFmpeg would be a better choice, but OpenCV works for basic streaming via GStreamer if configured.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            raise ValueError(f"Could not open VideoWriter for output: {output_path}")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
