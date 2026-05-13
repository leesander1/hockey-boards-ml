import argparse
import sys
import cv2

from src.ingestion.video_stream import VideoStream
from src.inference.model_runner import ModelRunner
from src.compositing.homography import AdCompositor
from src.output.video_writer import VideoWriter

def main():
    parser = argparse.ArgumentParser(description="NHL Hockey Ad Blocker & Tracking Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Source video feed (URL, RTMP, or local file)")
    parser.add_argument("--ad", type=str, help="Replacement ad image or video")
    parser.add_argument("--output", type=str, help="Output destination (local file)")
    parser.add_argument("--stream", type=str, help="Output RTMP stream destination")
    parser.add_argument("--mock-inference", action="store_true", help="Run with mock inference for testing plumbing")
    
    args = parser.parse_args()
    
    print(f"Initializing pipeline with source: {args.source}")
    
    # 1. Initialize Ingestion
    stream = VideoStream(args.source).start()
    
    # 2. Initialize Inference Models
    print("Loading models (Board Segmentation, Player Instance Segmentation)...")
    runner = ModelRunner()
    if args.mock_inference:
        runner.mock = True
        
    # 3. Initialize Compositor
    compositor = AdCompositor(ad_image_path=args.ad)
    
    # 4. Initialize Output
    output_path = args.output if args.output else "output.mp4"
    writer = VideoWriter(output_path, stream.fps, stream.width, stream.height)
    
    print("Pipeline started. Press Ctrl+C to stop.")
    
    frames_processed = 0
    try:
        while True:
            ret, frame = stream.read()
            if not ret:
                # Stream ended or queue empty
                if not stream.running:
                    print("End of video stream.")
                    break
                continue
                
            # Run inference
            board_mask = runner.get_board_mask(frame)
            player_mask = runner.get_player_mask(frame)
            
            # Composite ad
            composited = compositor.apply_ad(frame, board_mask, player_mask)
            
            # Write to output
            writer.write(composited)
            
            frames_processed += 1
            if frames_processed % 30 == 0:
                print(f"Processed {frames_processed} frames...")
                
    except KeyboardInterrupt:
        print("\nShutting down pipeline gracefully...")
    finally:
        # Cleanup resources
        stream.stop()
        writer.release()
        print(f"Finished processing. Saved {frames_processed} frames to {output_path}")

if __name__ == "__main__":
    main()
