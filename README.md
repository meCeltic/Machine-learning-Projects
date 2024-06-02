# Machine-learning-Projects
# 1.Smart System for Workplace Posture Monitoring using Deep Learning-based CNN    

Develop an intelligent system to monitor and analyze workplace posture in real-time using deep learning techniques. 
- Technologies Used: Python, OpenCV, TensorFlow, Keras, PyTorch, NumPy, Pandas
- Integrated real-time video processing to monitor and analyze posture continuously using surveillance footage
- Developed and optimized convolutional neural networks (CNNs) for detecting and classifying various workplace postures.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
7. [Training the Model](#training-the-model)
8. [Real-Time Video Processing](#real-time-video-processing)
9. [Results](#results)
## Project Overview
Develop an intelligent system to monitor and analyze workplace posture in real-time using deep learning techniques.

## Technologies Used
- Python
- OpenCV
- TensorFlow
- Keras
- PyTorch
- NumPy
- Pandas

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/meCeltic/Machine-learning-Projects/tree/master/Sitting-Posture-Recognition
    ```
2. Navigate to the project directory:
    ```sh
    cd Sitting-Posture-Recognition
    ```
3. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```
4. Install the required packages.
## Usage
1. **Data Preparation**: Place your annotated data in the `data` directory.
2. **Training the Model**:
    ```sh
    model.py
    ```
3. **Real-Time Monitoring**:
    ```sh
    posture_realtime.py
    ```
4. The system will start capturing video from your camera and display real-time posture analysis.

## Model Architecture
The CNN model used for posture detection is designed with the following layers:
- Convolutional layers
- MaxPooling layers
- Fully connected layers
- Softmax activation for output classification

## Data Collection and Preprocessing
- Data was collected from workplace surveillance footage.
- Images were annotated with posture labels.
- Preprocessing steps include resizing, normalization, and splitting into training/validation/test sets.

Training was performed on annotated images with various postures.

## Real-Time Video Processing
OpenCV was used to capture real-time video feed and overlay posture classification results using the trained CNN model.

## Results
The model achieved an accuracy of 90% on the test dataset. Below are some visualizations and example outputs from the real-time monitoring system.


# 2.Accident Detection System using Surveillance Cameras

Develop an automated system to detect traffic accidents in real-time using surveillance camera footage.
- Technologies Used: Python,YOLO,  OpenCV, TensorFlow, Keras, PyTorch, NumPy, Pandas
- Developed and fine-tuned convolutional neural networks (CNNs) for object detection and scene classification.
- Trained models using large datasets to improve detection accuracy and robustness.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
7. [Training the Model](#training-the-model)
8. [Real-Time Video Processing](#real-time-video-processing)
9. [Results](#results)

## Project Overview
Develop an automated system to detect traffic accidents in real-time using surveillance camera footage.

## Technologies Used
- Python
- YOLO
- OpenCV
- TensorFlow
- Keras
- PyTorch
- NumPy
- Pandas

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/accident-detection-system.git
    ```
2. Navigate to the project directory:
    ```sh
    cd accident-detection-system
    ```
3. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```
4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. **Data Preparation**: Place your annotated data in the `data` directory.
2. **Training the Model**:
    ```sh
    python train.py
    ```
3. **Real-Time Monitoring**:
    ```sh
    python monitor.py
    ```
4. The system will start capturing video from your camera and display real-time accident detection analysis.

## Model Architecture
The CNN model used for accident detection is designed with the following layers:
- Convolutional layers
- MaxPooling layers
- Fully connected layers
- Softmax activation for output classification

## Data Collection and Preprocessing
- Data was collected from surveillance footage of traffic scenes.
- Images were annotated with accident and non-accident labels.
- Preprocessing steps include resizing, normalization, and splitting into training/validation/test sets.

## Training the Model
The model was trained using:
- **YOLO** for object detection.
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

## Real-Time Video Processing
OpenCV was used to capture real-time video feed and overlay accident detection results using the trained YOLO model.

## Results
The model achieved an accuracy of 90% on the test dataset. Below are some visualizations and example outputs from the real-time monitoring system.

