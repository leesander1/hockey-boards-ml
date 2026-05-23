import cv2
import os

def main():
    video_path = "output/ml_composited.mp4"
    out_dir = "/Users/leesander/.gemini/antigravity-ide/brain/19cf1bf7-48df-4651-b795-7722108771ff/video_frames"
    os.makedirs(out_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
        
    frame_indices = [10, 50, 90, 130]
    current_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_idx in frame_indices:
            out_path = os.path.join(out_dir, f"frame_{current_idx:03d}.jpg")
            cv2.imwrite(out_path, frame)
            print(f"Saved {out_path}")
            
        current_idx += 1
        
    cap.release()
    print("Done extracting frames!")

if __name__ == "__main__":
    main()
