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
    # Grab frames over a much longer period of time (e.g., 900 frames = 30 seconds)
    # Take a frame every 30 frames to get a wide sequence.
    for i in range(1200):
        ret, frame = cap.read()
        if not ret: break
        if i % 30 == 0:
            # Scale down to speed up processing
            frame = cv2.resize(frame, (640, 360))
            frames.append(frame)
            
    cap.release()
    print(f"Extracted {len(frames)} frames across the video duration.")
    
    if len(frames) < 2:
        print("Not enough frames")
        return

    orb = cv2.ORB_create(nfeatures=3000)
    
    # We will stitch onto a much larger fixed canvas to hold the whole rink
    canvas_w = 4000
    canvas_h = 1500
    
    # Place the first image in the center
    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Center offset for the first image
    offset_x = (canvas_w - 640) // 2
    offset_y = (canvas_h - 360) // 2
    
    result[offset_y:offset_y+360, offset_x:offset_x+640] = frames[0]
    
    # The homography matrix of frames[0] relative to the canvas
    H_base = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)

    successful_stitches = 1
    last_good_frame = frames[0]

    for i in range(1, len(frames)):
        img1 = last_good_frame
        img2 = frames[i]
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Mask out the ice (bottom 60%) to only use features from boards/glass
        h, w = gray1.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:int(h*0.4), :] = 255 # Top 40% only
        
        kp1, des1 = orb.detectAndCompute(gray1, mask)
        kp2, des2 = orb.detectAndCompute(gray2, mask)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            print(f"Skipping frame {i}: Not enough features.")
            continue
            
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]
        
        if len(good_matches) < 10:
            print(f"Skipping frame {i}: Not enough good matches.")
            continue
            
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # M is affine transform from img2 to img1 (translation, rotation, scale only)
        # This prevents the 'smile' / U-shape keystoning distortion that findHomography causes on panning cameras.
        M, mask_h = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        
        if M is not None:
            # Convert 2x3 affine matrix to 3x3 homography matrix for warpPerspective
            M = np.vstack([M, [0, 0, 1]])
        else:
            print(f"Skipping frame {i}: Homography failed.")
            continue
            
        # Check if translation is absurdly large (which indicates a bad match)
        # M[0, 2] is X translation, M[1, 2] is Y translation
        if abs(M[0, 2]) > 400 or abs(M[1, 2]) > 200:
             print(f"Skipping frame {i}: Homography matrix looks unstable (jumped too far).")
             continue
             
        # To map img2 to the canvas, we multiply the base homography by M
        H_total = np.matmul(H_base, M)
        
        # Warp img2 directly to the large canvas
        warped_img2 = cv2.warpPerspective(img2, H_total, (canvas_w, canvas_h))
        
        # Simple blending using max (keeps the brightest pixels)
        result = np.maximum(result, warped_img2)
        
        # Update H_base for the next frame
        H_base = H_total
        last_good_frame = img2
        successful_stitches += 1
        print(f"Successfully stitched frame {i}/{len(frames)}")
        
    print(f"Total frames successfully stitched: {successful_stitches}")
        
    # Crop the canvas to remove the empty black borders
    gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_res, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        # Add a little padding
        pad = 20
        y1, y2 = max(0, y-pad), min(canvas_h, y+h+pad)
        x1, x2 = max(0, x-pad), min(canvas_w, x+w+pad)
        result = result[y1:y2, x1:x2]
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"Saved extended stitched image to {output_path}")

if __name__ == "__main__":
    # Process the longest video we have
    videos = sorted(glob.glob("src/*.mp4"))
    if not videos: exit(1)
    
    # We will pick the first video, but we can also process others if needed.
    # Let's try 21-12-18.mp4
    input_video = videos[0]
    output_image = "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/stitched_rink_extended.jpg"
    
    stitch_custom(input_video, output_image)
