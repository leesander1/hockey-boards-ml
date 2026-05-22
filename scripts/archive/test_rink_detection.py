import cv2
import numpy as np
import os
import glob

def find_rink_borders(video_path, output_path):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Skip to a frame that likely has a good view (e.g., 30th frame)
    for _ in range(30):
        ret, frame = cap.read()
        
    if not ret:
        print("Error: Could not read frame.")
        return

    print("Processing frame...")
    
    # 1. Convert to HSV for color thresholding
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. Define range for white/light-gray (the ice)
    # Ice in broadcast video isn't perfectly white due to lighting, shadows, and compression
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    
    # 3. Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 4. Morphological operations to remove noise (players, lines) and fill holes
    kernel = np.ones((15,15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 5. Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour, which we assume is the rink surface
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to a polygon to smooth out the borders
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 6. Draw the border on the original frame (Bright Red, thickness 5)
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 5)
        
        # Also draw some text
        cv2.putText(frame, "Identified Rink Borders", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        print("No contours found.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 7. Save the output
    success = cv2.imwrite(output_path, frame)
    if success:
        print(f"Successfully saved output to: {output_path}")
    else:
        print("Failed to save output.")

    cap.release()

if __name__ == "__main__":
    # Find the first mp4 file in src/
    videos = glob.glob("src/*.mp4")
    if not videos:
        print("No videos found in src/")
        exit(1)
        
    input_video = videos[0]
    output_image = "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/rink_test_output.jpg"
    
    find_rink_borders(input_video, output_image)
