import cv2
import glob

videos = sorted(glob.glob("src/*.mp4"))
for i, v in enumerate(videos):
    cap = cv2.VideoCapture(v)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames // 2) # get middle frame
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/video_{i}_middle.jpg", frame)
    cap.release()
