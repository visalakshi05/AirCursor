# 🖐️ Air Cursor – Hand Gesture Controlled Mouse

A real-time **virtual mouse system** built using **OpenCV, MediaPipe Tasks API, NumPy, and PyAutoGUI** that allows you to control your system cursor using hand gestures.

---

##  Features

- Real-time hand tracking using MediaPipe HandLandmarker
- Cursor movement using index fingertip
- Left click using thumb bend detection
- Right click using middle finger bend detection
- Live webcam visualization with landmark drawing
- Works in LIVE_STREAM mode for smooth interaction

---

## How It Works

### 1.Hand Detection
Uses the **MediaPipe Hand Landmarker Task Model** (`hand_landmarker.task`) to detect hand in real time.

### 2.Cursor Movement
- Index fingertip landmark (ID 8) is tracked.
- The normalized coordinates are mapped to screen resolution using `pyautogui`.
- X-axis is mirrored to match webcam orientation.

### 3.Gesture-Based Click Detection
Finger bending is detected using **angle calculation** between joints.

#### Left Click
- Triggered when **thumb angle < 140°**

#### Right Click
- Triggered when **middle finger angle < 140°**

Click state flags prevent multiple clicks from continuous bending.

---

## Technologies Used

- Python 3.x
- OpenCV
- MediaPipe Tasks API
- NumPy
- PyAutoGUI

---
