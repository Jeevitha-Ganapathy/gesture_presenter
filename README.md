#🎯 Gesture Presenter – Touchless PPT Controller

A gesture-based presentation controller using MediaPipe + OpenCV + cvzone.
Control slides, draw, erase, zoom, and use a laser pointer — all with hand gestures via webcam.

✨ Features

Slide navigation (Thumb → Prev, Pinky → Next)

Drawing with index finger

Erase strokes with 3 fingers

Laser pointer with 2 fingers

Zoom using pinch gesture

Live camera + slides side-by-side

⚙️ Installation
git clone https://github.com/your-username/gesture_presenter.git
cd gesture_presenter
pip install cvzone mediapipe opencv-python numpy


Place slides (images) inside the presentation/ folder.

▶️ Run
python gesture_presenter_cvzone.py

🎮 Gestures
Gesture	Action
👍 Thumb up	- Previous slide
🤙 Pinky up -	Next slide
☝️ Index up -	Draw
✊ Close fist	- End stroke
✌️ Index + Middle	- Laser pointer
🤟 3 fingers	- Erase stroke
🤏 Pinch -	Zoom
