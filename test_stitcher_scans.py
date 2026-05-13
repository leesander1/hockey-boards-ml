import cv2
import numpy as np

def test_stitcher():
    cap = cv2.VideoCapture("src/2026-05-12 21-12-18.mp4")
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Extract every 30th frame
        if i % 30 == 0:
            frames.append(cv2.resize(frame, (640, 360)))
        i += 1
    cap.release()
    print(f"Extracted {len(frames)} frames.")
    
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, pano = stitcher.stitch(frames)
    if status == cv2.Stitcher_OK:
        out_path = "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/video_stitch_scans.jpg"
        cv2.imwrite(out_path, pano)
        print("Success! Stitched video pano saved.")
    else:
        print(f"Stitching failed with status {status}")

if __name__ == "__main__":
    test_stitcher()
