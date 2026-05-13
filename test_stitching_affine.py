import cv2
import numpy as np
import glob
import os

def stitch_affine(video_path, output_path):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    frames = []
    # Grab 25 frames (about 12 seconds)
    for i in range(350):
        ret, frame = cap.read()
        if not ret: break
        if i % 14 == 0:
            frame = cv2.resize(frame, (800, 450))
            frames.append(frame)
            
    cap.release()
    print(f"Extracted {len(frames)} frames.")
    if len(frames) < 3: return

    sift = cv2.SIFT_create()
    
    # Precompute features for all frames, masking the ice out (bottom 50%)
    features = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        mask = np.ones_like(gray) * 255
        mask[int(gray.shape[0]*0.5):, :] = 0  # Mask bottom half (ice/players)
        kp, des = sift.detectAndCompute(gray, mask)
        features.append((kp, des))

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Center frame index
    center_idx = len(frames) // 2
    
    # Store transform from each frame to the center canvas
    # T_to_center[i] will be a 3x3 matrix
    T_to_center = [np.eye(3, dtype=np.float32) for _ in range(len(frames))]

    # Helper function to match and find affine transform from idx1 to idx2
    def get_transform(idx1, idx2):
        kp1, des1 = features[idx1]
        kp2, des2 = features[idx2]
        if des1 is None or des2 is None: return None
        
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
                    
        if len(good_matches) < 10: return None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Use partial affine (translation, rotation, scale) - avoids wild perspective warping
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None: return None
        
        # Convert 2x3 to 3x3
        M3 = np.eye(3, dtype=np.float32)
        M3[0:2, :] = M
        return M3

    # Calculate transforms from center outwards to right
    for i in range(center_idx + 1, len(frames)):
        M = get_transform(i, i-1) # Transform from i to i-1
        if M is not None:
            T_to_center[i] = np.matmul(T_to_center[i-1], M)
        else:
            print(f"Failed to match {i} to {i-1}")

    # Calculate transforms from center outwards to left
    for i in range(center_idx - 1, -1, -1):
        M = get_transform(i, i+1) # Transform from i to i+1
        if M is not None:
            T_to_center[i] = np.matmul(T_to_center[i+1], M)
        else:
            print(f"Failed to match {i} to {i+1}")

    canvas_w, canvas_h = 4000, 1500
    offset_x, offset_y = (canvas_w - 800) // 2, (canvas_h - 450) // 2
    
    H_base = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # To avoid harsh seams, we will blend using a distance transform or just simple alpha blending
    # Let's use a simple accumulator for blending
    accumulator = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_map = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

    # Create a feather mask for each frame to blend softly
    feather_mask = np.ones((450, 800, 1), dtype=np.float32)
    # Fade edges
    fade_px = 50
    for y in range(450):
        for x in range(800):
            dist_x = min(x, 800-x)
            dist_y = min(y, 450-y)
            dist = min(dist_x, dist_y)
            if dist < fade_px:
                feather_mask[y, x, 0] = dist / fade_px

    # Render frames (center last so it's on top / highest weight)
    # Actually, order doesn't matter with weight map addition
    for i, T in enumerate(T_to_center):
        # Final transform = H_base * T
        H_final = np.matmul(H_base, T)
        
        # Warp image
        warped_img = cv2.warpPerspective(frames[i], H_final, (canvas_w, canvas_h))
        # Warp mask
        warped_mask = cv2.warpPerspective(feather_mask, H_final, (canvas_w, canvas_h))
        warped_mask = np.expand_dims(warped_mask, axis=-1) if len(warped_mask.shape) == 2 else warped_mask
        
        accumulator += warped_img.astype(np.float32) * warped_mask
        weight_map += warped_mask

    # Normalize
    np.divide(accumulator, weight_map, out=result.astype(np.float32), where=weight_map > 0)
    result = result.astype(np.uint8)

    # Crop
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
        stitch_affine(videos[0], "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/stitched_affine.jpg")
