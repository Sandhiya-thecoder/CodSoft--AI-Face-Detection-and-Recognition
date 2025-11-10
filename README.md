<h1 align="center">👁️ Real-Time Face Detection & Recognition System</h1>

<p align="center">
A powerful <b>Python-based Face Detection and Recognition System</b> built using <b>OpenCV</b> and the <b>LBPH (Local Binary Patterns Histogram)</b> algorithm.  
Developed as my <b>final internship task</b>, this project demonstrates how computer vision enables machines to see, learn, and recognize human faces in real time. 🤖📸
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-green.svg" />
  <img src="https://img.shields.io/badge/State-Advanced-yellow.svg" />
  <img src="https://img.shields.io/badge/Status-Completed-success.svg" />
</p>

---

## 📘 Overview

This project focuses on **real-time human face detection and recognition** using a webcam feed.  
It applies **Haar Cascade Classifier** to locate faces in the frame and **LBPH algorithm** to identify the person by analyzing local pixel patterns.  

By combining **machine vision techniques** and **Python automation**, this system can capture, train, and recognize faces seamlessly.  
The entire process — from detection to recognition — happens in a fraction of a second, making it suitable for real-world applications like:
- Security authentication 🔐  
- Attendance systems 🧑‍💻  
- Access control systems 🏢  
- Smart surveillance systems 📹  

---

## 🧠 Tech Stack and Tools

| Component | Description |
|------------|-------------|
| 🐍 **Python** | Main programming language used for scripting and logic implementation |
| 👁️ **OpenCV** | Library for image processing, face detection, and recognition |
| 💾 **NumPy** | Used for array manipulation and matrix operations |
| 🧠 **LBPH (Local Binary Patterns Histogram)** | Algorithm for extracting local texture features from the face for accurate recognition |
| 🧩 **Haar Cascade Classifier** | Pre-trained XML-based model for detecting human faces from live video or images |
| 💻 **Webcam / Camera Feed** | Captures real-time frames for training and recognition |

---

## ⚙️ Working Mechanism

Here’s how the system works step by step:

1️⃣ **Face Detection**  
   The program activates your webcam and uses the **Haar Cascade Classifier** to detect faces in the live video feed. Detected faces are highlighted using rectangular bounding boxes.

2️⃣ **Training Mode**  
   When you press **‘t’**, the system begins capturing multiple samples of your face and stores them as image datasets. These serve as training data for the recognition model.

3️⃣ **Model Training**  
   Once enough samples are collected, the system trains a **Local Binary Patterns Histogram (LBPH)** model.  
   LBPH works by encoding facial textures into binary patterns and computing histograms that represent unique facial features.

4️⃣ **Recognition Mode**  
   After training, the webcam reopens for recognition.  
   When you press **‘q’**, the system begins comparing the **current live face** with the **trained model** (`trainer.yml`).  
   If a match is found, it displays **“Recognized”**; otherwise, it shows **“Different face.”**

5️⃣ **Output Display**  
   Recognition results are displayed live on the screen, providing instant visual feedback.

---

## 🧩 Algorithm Insight: Why LBPH?

LBPH (Local Binary Patterns Histogram) is one of the most reliable algorithms for face recognition because:
- It focuses on **texture patterns** rather than global features.  
- It performs exceptionally well under different **lighting conditions** and **facial expressions**.  
- It’s **fast, simple, and efficient** for real-time applications.  

This makes it a great choice for systems that require quick recognition without heavy computational load.

---

## 🖥️ Installation & Setup:
### import the following commands for installing libraries:
- pip install numpy opencv-python (numpy & cv)
- pip install opencv-contrib-python (face)

### 1️⃣ Clone the Repository
```bash
