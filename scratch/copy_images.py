import cv2
import shutil
import os

def main():
    workspace_img_dir = "images"
    os.makedirs(workspace_img_dir, exist_ok=True)
    
    brain_dir = "/Users/leesander/.gemini/antigravity-ide/brain/19cf1bf7-48df-4651-b795-7722108771ff"
    
    # 1. Copy walkthrough images (frame 10, 50, 90, 130)
    wt_frames = [10, 50, 90, 130]
    for f in wt_frames:
        src = os.path.join(brain_dir, "video_frames", f"frame_{f:03d}.jpg")
        dst = os.path.join(workspace_img_dir, f"frame_{f:03d}.jpg")
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print(f"Warning: Walkthrough frame not found at {src}")
            
    # 2. Copy purple viz image for frame 100
    src_purple = os.path.join(brain_dir, "purple_viz", "frame_100.jpg")
    dst_purple = os.path.join(workspace_img_dir, "purple_viz_frame_100.jpg")
    if os.path.exists(src_purple):
        shutil.copy(src_purple, dst_purple)
        print(f"Copied {src_purple} to {dst_purple}")
    else:
        print(f"Warning: Purple viz frame 100 not found at {src_purple}")
        
    # 3. Copy U-Net debug images from training validation
    val_files = [
        "val_unet_2026-05-19_23-25-44_f0630.jpg",
        "val_unet_2026-05-19_23-29-50_f0384.jpg"
    ]
    for idx, f in enumerate(val_files, 1):
        src = os.path.join(brain_dir, "validation_output", f)
        dst = os.path.join(workspace_img_dir, f"unet_val_{idx}.jpg")
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print(f"Warning: Validation debug image not found at {src}")
            
    # 4. Extract frame 100 of the final composited video
    video_path = "output/ml_composited.mp4"
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = cap.read()
        if ret:
            dst_comp = os.path.join(workspace_img_dir, "composite_frame_100.jpg")
            cv2.imwrite(dst_comp, frame)
            print(f"Successfully extracted frame 100 of composite video to {dst_comp}")
        else:
            print("Error: Could not read frame 100 from composite video")
        cap.release()
    else:
        print(f"Warning: Composite video not found at {video_path}")

if __name__ == "__main__":
    main()
