# Stitching NHL Rink Panoramas & Repo Cleanup

This plan addresses two parts: cleaning up the repository (organizing large files, updating documentation) and executing our initial plan to build a multi-stage panoramic stitching pipeline for NHL rink boards.

## User Review Required

> [!IMPORTANT]  
> Please review the `Open Questions` section below, specifically regarding how you want the panoramas saved and what input video format we should target for the sideline footage!

## Open Questions

1. **Panorama Output:** Do you want the final stitched panoramas saved as ultra-wide `.jpg` images in a dedicated `output/` folder, or do you want to keep them in memory for downstream ad-replacement directly?
2. **Input Videos:** We have a few `.mp4` files currently sitting in `src/`. Should we create a `data/videos/` folder for these and target them for the sideline stitching tests? 

## Proposed Changes

---

### Phase 1: Repository Cleanup & Documentation

To keep the repository clean and avoid accidentally committing massive files to Git, we'll perform the following:

#### [NEW] `.gitignore`
Add rules for ignoring large files:
- `*.mp4`, `*.mov`, `*.avi`
- `*.pth`, `*.pt` (PyTorch and YOLO weights)
- `colab_training_data.zip`
- `annotation_frames/`, `test_images/`, `tmp_frames/`
- Python caches (`__pycache__`, `.DS_Store`)

#### [MODIFY] `README.md`
Update the README to reflect the current state of the project:
- Add a section on **Board Segmentation** (how we detect boards).
- Add instructions on how to use Google Colab for fine-tuning our custom U-Net model.
- Add instructions for running validation checks.
- Document our upcoming **Panorama Stitcher** architecture.
- Add a note that HockeyAI semantic detections and HockeyRink keypoints can be used as geometry anchors to align board masks to known rink structure.

#### [DELETE] Temporary Scripts
We'll clean up the root directory by deleting:
- `organize_annotations.py`
- `update_ipynb.py`
- `update_ipynb_aug.py`
- Moving `.mp4` videos out of `src/` and into a designated `data/videos/` folder (or removing them if you have them backed up).

---

### Phase 2: Panoramic Stitching Pipeline

We will construct a new module to handle the construction of a high-fidelity panorama from multiple frames, specifically targeting the sideline cameras to overcome perspective distortion.

#### [NEW] `src/compositing/panorama_stitcher.py`
This module will execute the multi-stage stitching process:
1. **Feature Extraction:** Use `cv2.SIFT` to detect keypoints and compute descriptors across consecutive video frames.
2. **Matching & Homography:** Use `cv2.FlannBasedMatcher` and `cv2.findHomography` (with RANSAC) to stitch frames together sequentially, building a raw "continuous" wide image of the sideline boards.
3. **Template-based Geometry Alignment:** Because a purely sequential SIFT stitch accumulates perspective drift over time, we will feed the stitched result into our existing `rink_template.py` / `rink_calibrator.py` logic. This step forces the stitched "curved" panorama to flatten out and align exactly to the mathematical shape of an NHL rink.
4. **Rink Feature Anchors:** Use HockeyAI detections (`centerIce`, `faceoff`, `goal`, `player`, `puck`, `referee`) and HockeyRink keypoints as calibration anchors to estimate a frame-to-rink homography before board projection. If the HockeyRink keypoint order is not documented, infer a stable correspondence map from the model outputs and the rink template geometry.

#### [MODIFY] `src/inference/model_runner.py` (or similar orchestrator)
- Add an entry point to consume a video and run the `PanoramaStitcher` module to export the final stitched rink.

## Verification Plan

### Automated Tests
- We will process a short clip of low-angle sideline footage.
- Verify that SIFT matches correctly identify overlapping board segments.
- Verify that the final panorama output matches the mathematical dimensions expected by our `rink_template`.
- Verify that HockeyAI/HockeyRink-derived anchors recover a stable homography on frames with center ice and blue-line visibility.

### Manual Verification
- We'll output the raw SIFT stitch alongside the template-aligned stitch.
- I will embed the resulting images into the Walkthrough artifact so you can visually verify that there are no "tearing" artifacts and that the perspective drift has been corrected.
