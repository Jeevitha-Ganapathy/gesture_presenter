# 🎯 Gesture Presenter – Touchless PPT Controller  

This project is a **gesture-based presentation controller** built using **MediaPipe, OpenCV, and cvzone**. It allows you to control slides, draw annotations, erase strokes, zoom, and even use a laser pointer — all through simple **hand gestures** captured by your webcam. No clickers or keyboards are required, making presentations more intuitive and accessible.  

## ✨ Features  
- Navigate slides with gestures (Thumb → Previous, Pinky → Next)  
- Draw on slides using the index finger  
- Erase last stroke with three fingers  
- Use a laser pointer with index + middle fingers  
- Zoom in/out with a pinch gesture  
- Camera feed and slides are displayed side by side in real-time  

## ⚙️ Installation  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/Jeevitha-Ganapathy/gesture_presenter.git
cd gesture_presenter
pip install cvzone mediapipe opencv-python numpy

**🎮 Gestures**

👍 Thumb up → Previous slide

🤙 Pinky up → Next slide

☝️ Index up → Draw

✊ Close fist → End stroke

✌️ Index + Middle → Laser pointer

🤟 Three fingers → Erase stroke

🤏 Pinch (Thumb + Index) → Zoom
