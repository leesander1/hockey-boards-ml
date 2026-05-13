import cv2
import glob

def test_stitcher():
    image_files = sorted(glob.glob("src/Screenshot*.png"))
    images = [cv2.imread(f) for f in image_files if cv2.imread(f) is not None]
    
    # We need to order them geographically: Left, Center, Right.
    # The screenshots are:
    # 10.08.20 = Left
    # 10.08.49 = Center
    # 10.08.46 = Right
    images_ordered = [images[0], images[2], images[1]]
    
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, pano = stitcher.stitch(images_ordered)
    if status == cv2.Stitcher_OK:
        out_path = "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/stitched_3_photos_scans.jpg"
        cv2.imwrite(out_path, pano)
        print("Success! Stitched 3-photo pano saved.")
    else:
        print(f"Stitching failed with status {status}")

if __name__ == "__main__":
    test_stitcher()
