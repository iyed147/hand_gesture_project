# ✋ Hand Gesture Detection

> 🔗 This project is a companion module of the [Adaptive AI Pong Game](https://github.com/iyed147/Adaptive_AI_Pong_Game) — where hand gestures are used to control the paddle in real time.

---

## 📌 Overview

A real-time hand gesture recognition system built with **MediaPipe**, **OpenCV**, and **scikit-learn**.  
It detects hand landmarks via webcam and classifies gestures using a trained ML classifier.

---

## ⚙️ Installation

```bash
git clone https://github.com/iyed147/hand_gesture_project.git
cd hand_gesture_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Collect gesture data
```bash
python collect_data.py
```

### 2. Train the classifier
```bash
python train_classifier.py
```

### 3. Run real-time inference
```bash
python realtime_inference.py
```

---

## 🧰 Tech Stack

| Tool | Role |
|------|------|
| MediaPipe | Hand landmark detection |
| OpenCV | Webcam feed & visualization |
| scikit-learn | Gesture classification |
| NumPy | Data processing |

---

## 🔗 Related Project

This module is integrated into the **Adaptive AI Pong Game** —  
a Q-learning based Pong game controlled by hand gestures in real time.  
👉 [View the main project](https://github.com/iyed147/Adaptive_AI_Pong_Game)
