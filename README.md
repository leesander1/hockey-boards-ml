# NHL Hockey Ad Blocker & Real-Time Tracking

This project is a real-time machine learning pipeline that ingests a live NHL video feed, segments the rink boards, tracks players/objects (like the puck and referees), and composites a new ad (or blanks the boards) without occluding the foreground action.

## Architecture

1. **Ingestion (`src/ingestion`)**: Captures RTMP/HLS video streams or local video files via OpenCV and places frames into a high-performance multi-threaded queue.
2. **Inference (`src/inference`)**: Runs highly optimized deep learning models (e.g., custom U-Net segmentation networks) via TensorRT/PyTorch to generate pixel-perfect masks for the rink boards and the players.
3. **Compositing (`src/compositing`)**: Calculates homography (perspective transform) to warp new digital ads onto the board masks, and uses the player masks to preserve foreground elements over the new ads.
4. **Output (`src/output`)**: Streams the processed composited frames to an output sink (RTMP broadcast or local video file).
5. **Panoramic Stitcher (Upcoming)**: A multi-stage pipeline using SIFT and RANSAC to stitch multiple camera frames into a seamless high-fidelity panorama, corrected to the geometric template of an NHL rink.

## Prerequisites

- Python 3.10+
- NVIDIA GPU (RTX 30XX/40XX series recommended) with CUDA and cuDNN installed.
- PyTorch for model inference.
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

### Board Segmentation & Fine-Tuning

We use a custom lightweight U-Net architecture (`src/calibration/ml_board_detector.py`) to generate a binary mask of the rink boards. Because of varied broadcast quality and rink setups, you may need to fine-tune the model for specific scenarios (like heavy ice reflections).

**To fine-tune the model in Google Colab:**
1. Manually annotate some challenging frames in `annotation_frames/new_batch/annotated/`. 
   - Add the original `frame.jpg`, the colored `frame-annotated.png`, and the binary `frame-mask.png`.
2. Package the dataset by running:
   ```bash
   python prepare_colab_upload.py
   ```
   This will generate a `colab_training_data.zip`.
3. Open `train_board_segmentation_colab.ipynb` in Google Colab.
4. Ensure your Colab runtime is set to **T4 GPU**.
5. Upload the `colab_training_data.zip` file when prompted and run all cells.
6. Download the resulting `board_segmentation_model.pth` and place it in `src/calibration/`.

### Validating the Model

To ensure your fine-tuned model performs well and rejects noise (like ice reflections and crowd anomalies), run the validation script:

```bash
python validate_model.py
```

This will run the updated model against random frames in `annotation_frames/new_batch/` and output visual side-by-side comparisons (including probability heatmaps and final masked overlays) into `test_images/validation_output/`.

### Running the Pipeline
To run the ad-blocker pipeline on a local video file:
```bash
python main.py --source data/videos/input.mp4 --ad data/replacement_ad.jpg --output data/output.mp4
```

To optionally bias the board mask with external rink features, pass a HockeyAI detector and/or a HockeyRink pose model:
```bash
python main.py --source data/videos/input.mp4 --ad data/replacement_ad.jpg --output data/output.mp4 \
   --hockeyai-model path/to/hockeyai.pt --hockeyrink-model path/to/hockeyrink.pt
```
HockeyRink keypoints still need a keypoint-to-world mapping in code because the public model card does not publish the 56-point order.
If you have that mapping saved in JSON, pass it with `--hockeyrink-keypoint-map path/to/map.json`.

### Notes
- Auxiliary feature sources that may improve board localization:
   - https://huggingface.co/datasets/SimulaMet-HOST/HockeyAI for player and other class labels
   - https://huggingface.co/SimulaMet-HOST/HockeyRink for rink masks
   - https://huggingface.co/Edalik/hockey for additional hockey imagery/data
- You can also get the ice mask from OpenCV in some broadcasts (ice is white, everything else is darker).
- For board detection and alignment, use the yellow kickplate / trim lines as a geometric anchor.

