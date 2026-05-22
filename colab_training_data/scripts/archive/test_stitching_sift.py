import cv2
import numpy as np
import glob
import os

def stitch_custom(video_path, output_path):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    frames = []
    # Grab fewer frames over a shorter period to avoid infinite drift
    # Let's take 20 frames spaced by 15 (about 10 seconds of video)
    for i in range(300):
        ret, frame = cap.read()
        if not ret: break
        if i % 15 == 0:
            frame = cv2.resize(frame, (800, 450))
            frames.append(frame)
            
    cap.release()
    print(f"Extracted {len(frames)} frames.")
    
    if len(frames) < 2:
        return

    # Use SIFT instead of ORB for better feature matching
    sift = cv2.SIFT_create()
    
    # Large canvas
    canvas_w = 4000
    canvas_h = 1500
    
    # We will compute homographies relative to the FIRST frame for simplicity,
    # but we'll chain them carefully and stop if error is high.
    
    offset_x = (canvas_w - 800) // 2
    offset_y = (canvas_h - 450) // 2
    
    H_base = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # We will accumulate aligned images here to take a median later
    # This prevents the "white blur" from maximum blending
    aligned_frames = []
    
    # Start with first frame
    warp0 = cv2.warpPerspective(frames[0], H_base, (canvas_w, canvas_h))
    aligned_frames.append(warp0)

    last_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(last_gray, None)

    # Use Flann for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for i in range(1, len(frames)):
        img2 = frames[i]
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            continue
            
        matches = flann.knnMatch(des2, des1, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                
        if len(good_matches) < 15:
            print(f"Skipping frame {i}: Not enough good matches ({len(good_matches)}).")
            # We don't update last_gray, we just try matching the next frame to the last good one
            continue
            
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            continue
            
        # Update H_base
        H_base = np.matmul(H_base, M)
        
        warped_img2 = cv2.warpPerspective(img2, H_base, (canvas_w, canvas_h))
        aligned_frames.append(warped_img2)
        
        # Update features for next iteration
        kp1, des1 = kp2, des2
        print(f"Successfully aligned frame {i}/{len(frames)}")

    print(f"Blending {len(aligned_frames)} aligned frames...")
    
    # Blend using "last frame on top" or something simpler than max to avoid white smearing
    # Let's just overlay them sequentially, replacing non-black pixels
    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for frame in aligned_frames:
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 0
        result[mask] = frame[mask]
        
    # Crop borders
    gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_res, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        pad = 20
        y1, y2 = max(0, y-pad), min(canvas_h, y+h+pad)
        x1, x2 = max(0, x-pad), min(canvas_w, x+w+pad)
        result = result[y1:y2, x1:x2]
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"Saved stitched image to {output_path}")

if __name__ == "__main__":
    videos = sorted(glob.glob("src/*.mp4"))
    if videos:
        stitch_custom(videos[0], "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/stitched_sift.jpg")
