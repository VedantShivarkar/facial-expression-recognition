# Real-Time Facial Expression Recognition System

This project is a deep learning-based computer vision application that detects human faces in real-time via a webcam and classifies their emotional state into one of seven categories. It was developed as an academic assessment project.

**Author:** Vedant Shivarkar  
**Institution:** Yeshwantrao Chavan College of Engineering  

## 🎯 Project Overview
The system utilizes a Convolutional Neural Network (CNN) trained on the FER-2013 (Facial Expression Recognition) dataset. For real-time inference, it leverages OpenCV's Haar Cascades to isolate faces in a live video stream, preprocesses the region of interest (ROI), and feeds it to the trained CNN to predict the emotion alongside a confidence percentage.

### Detectable Emotions
* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

## ⚙️ Tech Stack & Tools
* **Programming Language:** Python 3.x
* **Deep Learning Framework:** TensorFlow / Keras (CNN Architecture)
* **Computer Vision:** OpenCV (Face detection and live camera feed)
* **Data Processing:** NumPy
* **Development Environment:** Google Colab (Model Training on T4 GPU) & Visual Studio Code (Local Inference)

## 📁 Project Structure
```text
Facial_Expression_Project/
│
├── model/
│   ├── emotion_model.h5                     # Trained CNN model
│   └── haarcascade_frontalface_default.xml  # OpenCV pre-trained face detector
│
├── real_time_detection.py                   # Main script for real-time webcam inference
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation

🚀 Setup and Installation
1. Prerequisites
Ensure you have Python installed on your system. It is recommended to use a virtual environment.

2. Install Dependencies
Navigate to the project directory in your terminal and install the required libraries:

Bash
pip install -r requirements.txt
(The requirements.txt should contain: opencv-python, tensorflow, and numpy)

3. Model Files
Ensure that both emotion_model.h5 and haarcascade_frontalface_default.xml are placed inside the model/ directory.

💻 How to Run
Open a terminal in the root directory of the project.

Execute the following command:

Bash
python real_time_detection.py
A window named "Emotion Detector" will open, activating your webcam.

The system will draw bounding boxes around detected faces and display the predicted emotion along with its confidence score (e.g., Happy: 98.5%).

Press 'q' to quit the application and close the window.

🧠 Model Architecture details
The CNN was built sequentially with the following structure to process 48x48 pixel grayscale images:

3 Convolutional Blocks: Each containing Conv2D layers, BatchNormalization, MaxPooling2D, and Dropout (0.25) to extract spatial features and prevent overfitting.

Fully Connected Layer: A Flatten layer followed by a Dense layer of 512 units (ReLU) and Dropout (0.5).

Output Layer: A Dense layer with 7 units and a softmax activation function for multi-class classification.

Optimization: Compiled using the Adam optimizer and categorical_crossentropy loss function.
