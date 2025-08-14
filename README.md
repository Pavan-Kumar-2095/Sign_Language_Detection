# 🤖 Real-Time Hand Gesture to Text Translator using Deep Learning

This project is a **real-time sign language recognition system** that translates hand gestures into readable text using deep learning and computer vision.

Built with **MediaPipe**, **OpenCV**, and a custom-trained **Artificial Neural Network (ANN)**, the system processes hand landmarks in real time and converts them into meaningful words.

---

## 🚀 How It Works

1. **MediaPipe** captures hand landmarks from a live video stream.
2. **OpenCV** handles video frame processing and display.
3. A trained **ANN model** classifies gestures based on hand keypoints.
4. Recognized gestures are continuously processed and assembled into full words.

---

## 🧠 Tech Stack

| Tool/Library      | Role                                   |
|-------------------|----------------------------------------|
| 📹 MediaPipe       | Real-time hand tracking                |
| 🖼️ OpenCV          | Video feed capture and preprocessing   |
| 🐍 Python           | Core logic and integration             |
| 🧠 TensorFlow/Keras | ANN model training and prediction      |
| 🗂️ Custom Dataset   | Hand-collected for gesture accuracy   |

---

## ⚡ Why ANN?

- Efficient classification of gesture data
- Low memory usage and fast inference
- Ideal for **real-time applications** on limited hardware

---

## 📂 Dataset

The dataset used for training was **hand-collected** to ensure:
- Realistic gesture variations
- Accurate hand landmark labeling
- Consistent performance across different lighting/backgrounds

> Optionally: You can include `data/` samples or a notebook for data collection if open-sourcing the dataset.
