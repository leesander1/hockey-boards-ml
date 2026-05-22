import cv2
import numpy as np
import os
import glob

def stitch_video_frames(video_path, output_path, max_frames=15, skip_frames=30):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frames = []
    frame_count = 0
    collected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % skip_frames == 0:
            # Resize frame slightly to speed up stitching
            height, width = frame.shape[:2]
            scale = 0.5
            resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
            frames.append(resized)
            collected_count += 1
            print(f"Collected frame {collected_count}")
            
            if collected_count >= max_frames:
                break
                
        frame_count += 1

    cap.release()

    if len(frames) < 2:
        print("Not enough frames to stitch.")
        return

    print(f"Stitching {len(frames)} frames together...")
    
    # Use OpenCV's built-in stitcher (PANORAMA mode works well for panning cameras)
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, stitched = stitcher.stitch(frames)

    if status == cv2.Stitcher_OK:
        print("Stitching successful!")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, stitched)
        print(f"Saved stitched image to: {output_path}")
    else:
        print(f"Stitching failed with status code: {status}")
        # Status 1 = ERR_NEED_MORE_IMGS, 2 = ERR_HOMOGRAPHY_EST_FAIL, 3 = ERR_CAMERA_PARAMS_ADJUST_FAIL

if __name__ == "__main__":
    # Get the longest/best video for panning
    videos = sorted(glob.glob("src/*.mp4"))
    if not videos:
        print("No videos found in src/")
        exit(1)
        
    # We will just pick the first video for now
    input_video = videos[0]
    output_image = "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/stitched_rink.jpg"
    
    # Try grabbing a frame every 15 frames, up to 20 frames total
    stitch_video_frames(input_video, output_image, max_frames=20, skip_frames=15)
