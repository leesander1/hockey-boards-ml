import cv2
import glob
import os

def main():
    # Find the 3 screenshots
    image_files = sorted(glob.glob("src/Screenshot*.png"))
    
    if len(image_files) != 3:
        print(f"Found {len(image_files)} screenshots, expected 3.")
        return

    print("Loading images...")
    images = []
    for file in image_files:
        img = cv2.imread(file)
        if img is not None:
            images.append(img)
            print(f"Loaded {file}")
        else:
            print(f"Failed to load {file}")

    if len(images) != 3:
        print("Could not load all 3 images.")
        return

    print("Stitching images...")
    
    # Use OpenCV's Stitcher class
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, stitched = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("Stitching successful!")
        output_path = "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/stitched_3_photos.jpg"
        cv2.imwrite(output_path, stitched)
        print(f"Saved stitched panorama to {output_path}")
    else:
        print(f"Stitching failed with status code: {status}")
        # Status codes:
        # 1: ERR_NEED_MORE_IMGS
        # 2: ERR_HOMOGRAPHY_EST_FAIL
        # 3: ERR_CAMERA_PARAMS_ADJUST_FAIL

if __name__ == "__main__":
    main()
