import cv2

cap = cv2.VideoCapture("src/2026-05-12 21-12-18.mp4")
ret, frame0 = cap.read()
if ret:
    cv2.imwrite("frame_0.jpg", frame0)

# get last frame
cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
ret, frame_last = cap.read()
if ret:
    cv2.imwrite("frame_last.jpg", frame_last)
cap.release()
print("Saved frame_0.jpg and frame_last.jpg")
