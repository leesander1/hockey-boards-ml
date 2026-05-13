# NHL Hockey Ad Blocker & Real-Time Tracking

This project is a real-time machine learning pipeline that ingests a live NHL video feed, segments the rink boards, tracks players/objects (like the puck and referees), and composites a new ad (or blanks the boards) without occluding the foreground action.

## Architecture

1. **Ingestion (`src/ingestion`)**: Captures RTMP/HLS video streams or local video files via OpenCV and places frames into a high-performance multi-threaded queue.
2. **Inference (`src/inference`)**: Runs highly optimized deep learning models (e.g., YOLOv8-Seg or custom segmentation networks) via TensorRT to generate pixel-perfect masks for the rink boards and the players.
3. **Compositing (`src/compositing`)**: Calculates homography (perspective transform) to warp new digital ads onto the board masks, and uses the player masks to preserve foreground elements over the new ads.
4. **Output (`src/output`)**: Streams the processed composited frames to an output sink (RTMP broadcast or local video file).

## Prerequisites

- Python 3.10+
- NVIDIA GPU (RTX 30XX/40XX series recommended) with CUDA and cuDNN installed.
- TensorRT for model acceleration.
- FFmpeg (for streaming output).

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd hockey-ad-blocker
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Pipeline
To run the ad-blocker pipeline on a local video file:
```bash
python main.py --source data/input.mp4 --ad data/replacement_ad.jpg --output data/output.mp4
```

To run on a live stream (e.g., RTMP):
```bash
python main.py --source rtmp://live-feed-url --ad data/replacement_ad.mp4 --stream rtmp://output-feed-url
```

### Testing
Run unit tests and performance benchmarks:
```bash
pytest tests/
```

### Deployment
To deploy this as a continuous service, we recommend wrapping the pipeline in a Docker container with NVIDIA Container Toolkit support. A basic deployment command:
```bash
docker build -t hockey-ad-blocker .
docker run --gpus all -v $(pwd)/data:/app/data hockey-ad-blocker --source rtmp://...
```
