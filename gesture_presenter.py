# gesture_presenter_cvzone.py
# Requires: cvzone, mediapipe, opencv-python, numpy
# Install once with:
#   pip install cvzone mediapipe opencv-python numpy

from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np
import math
import time

# ---------------- PARAMETERS ----------------
width, height = 640, 480             # camera feed size
slideWidth, slideHeight = 960, 720   # slide display size
gestureThreshold = 240               # y threshold in camera coords to enable slide nav
folderPath = r"C:\Users\jeevi\gesture-presenter\presentation"  # <-- your slides folder

# smoothing / behavior params
INDEX_EMA_ALPHA = 0.25
PINCH_HYSTERESIS = 8
PINCH_MIN = 20
PINCH_MAX = 120
ZOOM_MIN = 1.0
ZOOM_MAX = 3.0
ZOOM_SMOOTH_ALPHA = 0.2
DRAW_MIN_MOVE = 4
SWIPE_DEBOUNCE_FRAMES = 20
ERASE_DEBOUNCE_FRAMES = 20
FPS_SMOOTH = 0.9

# ---------------- SETUP ----------------
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Load slides list
pathImages = sorted([f for f in os.listdir(folderPath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
if not pathImages:
    raise SystemExit(f"No slide images found in {folderPath}")

# state
imgNumber = 0
annotations = [[]]
annotationNumber = 0
zoomScale = 1.0
zoomTarget = 1.0
zoomCenter = (slideWidth // 2, slideHeight // 2)
pinch_active = False
pinch_start_dist = None
last_erase_time = -9999
last_nav = -9999
index_x_ema, index_y_ema = None, None
fps = 0.0
tprev = time.time()

# helper to clamp
def clamp(x, a, b): 
    return max(a, min(b, x))

print("Slides found:", pathImages)

while True:
    tnow = time.time()
    dt = tnow - tprev
    tprev = tnow
    if dt > 0:
        fps_inst = 1.0 / dt
        fps = FPS_SMOOTH * fps + (1 - FPS_SMOOTH) * fps_inst

    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    # load slide
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgSlide = cv2.imread(pathFullImage)
    if imgSlide is None:
        print("Failed to load", pathFullImage)
        break
    imgSlide = cv2.resize(imgSlide, (slideWidth, slideHeight))

    # detect hand
    hands, img = detectorHand.findHands(img, flipType=False)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 2)
    gesture_text = "No hand"

    if hands:
        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]
        fingers = detectorHand.fingersUp(hand)

        # index finger coords
        raw_ix_cam, raw_iy_cam = lmList[8][0], lmList[8][1]
        if index_x_ema is None:
            index_x_ema, index_y_ema = raw_ix_cam, raw_iy_cam
        else:
            index_x_ema = int((1 - INDEX_EMA_ALPHA) * index_x_ema + INDEX_EMA_ALPHA * raw_ix_cam)
            index_y_ema = int((1 - INDEX_EMA_ALPHA) * index_y_ema + INDEX_EMA_ALPHA * raw_iy_cam)

        ix = int(np.interp(index_x_ema, [0, width], [0, slideWidth]))
        iy = int(np.interp(index_y_ema, [0, height], [0, slideHeight]))
        indexFinger = (ix, iy)

        now = time.time()
        # -------- Slide navigation --------
        if cy <= gestureThreshold:
            if fingers == [1, 0, 0, 0, 0] and (now - last_nav) > SWIPE_DEBOUNCE_FRAMES / 30.0:
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = 0
                    last_nav = now
                    gesture_text = "PREV SLIDE"
            elif fingers == [0, 0, 0, 0, 1] and (now - last_nav) > SWIPE_DEBOUNCE_FRAMES / 30.0:
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = 0
                    last_nav = now
                    gesture_text = "NEXT SLIDE"

        # -------- Drawing --------
        if fingers == [0, 1, 0, 0, 0]:
            gesture_text = "DRAW"
            if annotationNumber >= len(annotations):
                annotations.append([])
            if not annotations[annotationNumber]:
                annotations[annotationNumber].append(indexFinger)
            else:
                prev_pt = annotations[annotationNumber][-1]
                dist = math.hypot(indexFinger[0] - prev_pt[0], indexFinger[1] - prev_pt[1])
                if dist >= DRAW_MIN_MOVE:
                    annotations[annotationNumber].append(indexFinger)
        elif fingers == [0, 0, 0, 0, 0]:
            if annotations and annotations[annotationNumber]:
                annotationNumber += 1
                annotations.append([])

        # -------- Erase last stroke --------
        if fingers == [0, 1, 1, 1, 0]:
            if (now - last_erase_time) > (ERASE_DEBOUNCE_FRAMES / 30.0):
                for i in range(len(annotations) - 1, -1, -1):
                    if annotations[i]:
                        annotations.pop(i)
                        annotationNumber = max(0, len(annotations) - 1)
                        break
                last_erase_time = now
                gesture_text = "ERASE"

        # -------- Laser pointer --------
        if fingers == [0, 1, 1, 0, 0]:
            gesture_text = "LASER"
            cv2.circle(imgSlide, indexFinger, 15, (0, 0, 255), cv2.FILLED)

        # -------- Zoom (pinch) --------
        x1, y1 = lmList[4][0], lmList[4][1]
        x2, y2 = lmList[8][0], lmList[8][1]
        pinch_len = math.hypot(x2 - x1, y2 - y1)

        if not pinch_active and pinch_len < PINCH_MIN:
            pinch_active = True
            pinch_start_dist = pinch_len
            zoomCenter = indexFinger
        elif pinch_active and pinch_len > PINCH_MAX:
            pinch_active = False
            pinch_start_dist = None

        if pinch_active and pinch_start_dist and pinch_start_dist > 0:
            ratio = pinch_start_dist / max(pinch_len, 1)
            target = clamp(ratio, ZOOM_MIN, ZOOM_MAX)
            zoomTarget = target
            zoomScale = (1 - ZOOM_SMOOTH_ALPHA) * zoomScale + ZOOM_SMOOTH_ALPHA * zoomTarget
            gesture_text = f"ZOOM {zoomScale:.2f}"
        else:
            zoomTarget = 1.0
            zoomScale = (1 - ZOOM_SMOOTH_ALPHA) * zoomScale + ZOOM_SMOOTH_ALPHA * zoomTarget

    # -------- Apply zoom --------
    outSlide = imgSlide.copy()
    if zoomScale > 1.001:
        cx, cy = zoomCenter
        w = int(slideWidth / zoomScale)
        h = int(slideHeight / zoomScale)
        x1 = clamp(cx - w // 2, 0, slideWidth - w)
        y1 = clamp(cy - h // 2, 0, slideHeight - h)
        x2 = int(x1 + w)
        y2 = int(y1 + h)
        imgCrop = imgSlide[y1:y2, x1:x2]
        outSlide = cv2.resize(imgCrop, (slideWidth, slideHeight))

    # -------- Draw annotations --------
    for stroke in annotations:
        if len(stroke) >= 2:
            for i in range(1, len(stroke)):
                cv2.line(outSlide, stroke[i-1], stroke[i], (0, 0, 255), 6)

    # -------- Combine camera + slide --------
    imgCameraSmall = cv2.resize(img, (width, height))
    H = max(height, slideHeight)
    W = width + slideWidth
    imgCombined = np.zeros((H, W, 3), np.uint8)
    imgCombined[0:height, 0:width] = imgCameraSmall
    imgCombined[0:slideHeight, width:width + slideWidth] = outSlide

    cv2.putText(imgCombined, f"FPS: {fps:.1f}", (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(imgCombined, f"Slide {imgNumber + 1}/{len(pathImages)}   Gesture: {gesture_text}",
                (width + 10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    instructions = [
        "Above green line: raise hand to change slides",
        "Thumb up -> Prev | Pinky up -> Next",
        "Index up -> Draw (close hand to finish stroke)",
        "Index+Mid up -> Laser",
        "Index+Mid+Ring up -> Erase last stroke",
        "Pinch Thumb+Index -> Zoom (center locked at pinch start)"
    ]
    for i, line in enumerate(instructions):
        cv2.putText(imgCombined, line, (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Camera & Slides", imgCombined)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        if imgNumber < len(pathImages) - 1:
            imgNumber += 1
            annotations = [[]]
            annotationNumber = 0
    elif key == ord('p'):
        if imgNumber > 0:
            imgNumber -= 1
            annotations = [[]]
            annotationNumber = 0

cap.release()
cv2.destroyAllWindows()
