# **Face Detection & Recognition System (Python + OpenCV)**

This project is a **Face Detection and Recognition System** built with **Python** and **OpenCV**.  
It uses a **Haar Cascade XML classifier** for detecting faces and a custom-trained model for face recognition.

The model in this project has been trained on:

- **H.E John Mahama**
- **H.E Nana Akufo-Addo**
- **Donald Trump**
- **Barack Obama**

Training generates two NumPy files:

- `features.npy` – extracted face features
- `labels.npy` – label indices for each face sample

A directory of training images is required for the program to train or retrain correctly.

---

## 📌 **Features**

- Real-time **face detection** using Haar Cascade.
- **Face recognition** using OpenCV’s LBPHFaceRecognizer.
- Training pipeline that extracts and labels face samples.
- Includes generated dataset files (`features.npy`, `labels.npy`).
- Easy to add more people by adding folders and retraining.

---

## 🛠️ **Technologies Used**

- **Python 3**
- **OpenCV**
- **NumPy**
- **Haar Cascade Classifier**

---

## 📁 **Project Structure**

project/
│── train.py
│── detect.py
│── features.npy
│── labels.npy
│── haarcascade_frontalface_default.xml
│── images/
│ ├── Mahama/
│ ├── Akufo-Addo/
│ ├── Trump/
│ └── Obama/
│── README.md
