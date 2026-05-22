import cv2
import glob

videos = glob.glob("src/*.mp4")
for v in videos:
    cap = cv2.VideoCapture(v)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{v}: {frames} frames")
    cap.release()
