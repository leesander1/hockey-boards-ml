import cv2
import os
import glob

# Search for videos in data/videos/
videos = glob.glob('data/videos/*.mp4')
output_dir = 'annotation_frames/new_batch/unannotated'
annotated_dir = 'annotation_frames/new_batch/annotated'
os.makedirs(output_dir, exist_ok=True)

frames_per_video = 15  # Extract 15 frames per video (user said "like 10 frames per video or more")

# Check existing files in both unannotated and annotated dirs to avoid extracting duplicates
existing_unannotated = os.listdir(output_dir) if os.path.exists(output_dir) else []
existing_annotated = os.listdir(annotated_dir) if os.path.exists(annotated_dir) else []

all_existing_files = existing_unannotated + existing_annotated

extracted_count = 0

for video_path in sorted(videos):
    # Format the name just like the existing ones (replacing spaces with underscores)
    vid_name = os.path.basename(video_path).split('.')[0].replace(' ', '_')
    
    # Check if we already have frames for this video
    prefix = f"{vid_name}_f"
    already_extracted = any(f.startswith(prefix) for f in all_existing_files)
    
    if already_extracted:
        print(f"Skipping {video_path} (frames already extracted)")
        continue
        
    print(f"Processing {video_path}...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"Warning: {video_path} has 0 frames. Skipping.")
        cap.release()
        continue
        
    # We want to pull frames evenly spaced, avoiding the very beginning and end if possible
    step = total_frames // (frames_per_video + 1)
    if step <= 0:
        step = 1
        
    for i in range(1, frames_per_video + 1):
        frame_idx = i * step
        if frame_idx >= total_frames:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"{vid_name}_f{frame_idx:04d}.jpg")
            cv2.imwrite(out_path, frame)
            print(f"  Extracted {out_path}")
            extracted_count += 1
            
    cap.release()

print(f"\nDone! Extracted {extracted_count} frames.")

