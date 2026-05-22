import os
import subprocess
import time

videos = [
    "2026-05-12 21-12-18.mp4",
    "2026-05-12 21-12-59.mp4",
    "2026-05-12 21-15-46.mp4",
    "2026-05-12 21-16-10.mp4",
    "2026-05-19 23-23-08.mp4",
    "2026-05-19 23-23-35.mp4",
    "2026-05-19 23-24-11.mp4",
]

model_path = "src/calibration/board_segmentation_model_unet.pth"
video_dir = "data/videos"
output_dir = "output"

os.makedirs(output_dir, exist_ok=True)

print("Starting batch processing of videos using UNet...")
for video_name in videos:
    video_path = os.path.join(video_dir, video_name)
    if not os.path.exists(video_path):
        print(f"Skipping {video_name} - not found.")
        continue
        
    output_name = f"unet_composited_{video_name.replace(' ', '_')}"
    output_path = os.path.join(output_dir, output_name)
    
    print(f"\n==========================================")
    print(f"Processing: {video_name}")
    print(f"Output:     {output_path}")
    print(f"==========================================")
    
    cmd = [
        "python3", "scripts/replace_boards_ml.py",
        "--video", video_path,
        "--model", model_path,
        "--output", output_path,
        "--max-frames", "150"
    ]
    
    start = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start
    
    if res.returncode == 0:
        print(f"Successfully processed {video_name} in {elapsed:.1f}s")
    else:
        print(f"Failed to process {video_name} (exit code: {res.returncode})")

print("\nBatch processing completed.")
