# AiLab Project Roadmap

We aim to develop a facial emotion recognition model capable of identifying a person's emotions. Following the training of the model, we plan to create a system for real-time facial recognition and simultaneous recognition of multiple people from an image.

## Roadmap

_Refer to the [README.md](README.md) file for dependency installation instructions._

### 1. Dataset Selection

We have chosen the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013). This version is smaller than the original and has balancing issues, but it is freely available and includes both images and a CSV file.

### 2. Balancing the Dataset

Ensure that the dataset is balanced. The chosen dataset has balancing issues, so it needs to be balanced.

| ![](https://i.ibb.co/ThjTVhC/dataset-graph.png) | ![](https://i.ibb.co/CPTDPhd/chart.png) |
|-------------------------------------------------|-----------------------------------------|

To balance the dataset, we can use various techniques such as oversampling, undersampling, class weights, etc.

### 3. Feature Extraction from Images

For feature extraction, we can use various techniques. The general idea is to use OpenCV for extracting keypoints with the algorithms discussed, neural networks, and/or machine learning algorithms. Alternatively, we can use the CSV file with pre-extracted data.

### 4. Model Training

For model training, we will use **scikit-learn** and **PyTorch**. We can experiment with different algorithms such as Support Vector Machine (SVM), Random Forest, and Artificial Neural Networks (ANN).

### 5. Model Evaluation and Validation

Evaluate the model's performance by calculating metrics like accuracy, precision, recall, and F1-score.

### 6. Model Optimization

Experiment with different machine learning algorithms, parameters, and feature extraction techniques such as grid search and random search to find the best parameters.

### 7. Model Application

At this stage, our model should be functioning, and the primary objectives of the project should be achieved.

## Side Quest

As mentioned, we aim to develop a system for real-time facial recognition and recognizing multiple people simultaneously from an image.

### 1. Real-time Recognition

For real-time facial recognition, we can use **OpenCV**, which allows us to open a window with the webcam and extract frames. We will also use **OpenCV** for facial recognition, drawing rectangles around detected faces.

### 1.1 Facial Recognition

For facial recognition, we use **OpenCV** to extract frames from the camera and recognize faces in the images. Here’s a simple guide on how to do it: [Face Detection with OpenCV](https://www.datacamp.com/tutorial/face-detection-python-opencv). We might also consider **dlib** if it proves to be more powerful.

### 2. Multiple Face Recognition from an Image

Using the previously mentioned method, we can simultaneously recognize multiple faces in the same image with **OpenCV**. This involves extracting frames from the camera and detecting faces in those frames.
