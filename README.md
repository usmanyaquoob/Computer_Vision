# Computer Vision Lab Tasks
This repository encompasses a set of tasks within the realm of computer vision labs, undertaken as components of coursework or projects. These assignments span diverse concepts and techniques in computer vision, showcasing adeptness in image processing, object detection, and related domains.

## Lab 01: Fundamentals of Image Processing


### Image Processing with OpenCV
- Loading and displaying images
- Converting images to grayscale
- Resizing images using OpenCV
- Drawing basic shapes on images
- Applying Gaussian blur, cropping, and manipulating images
- Adding text to images
- Applying binary thresholding and rotation
- Blending two images, converting to grayscale, and applying histogram equalization
- Performing bitwise operations on binary images
- Converting image pixel values to a Pandas DataFrame and applying masks

## Lab 02: Exploration of Datasets and Preprocessing

### Dataset Exploration
Exploring a dataset containing images of pets categorized into four classes: Angry, Sad, Happy, and Others. Displaying the number of samples in each class.

### Loading and Preprocessing Dataset
Loading the pet emotions dataset, resizing images, normalizing pixel values, and splitting the dataset into training and testing sets.

### Exploratory Data Analysis (EDA)
Conducting EDA on the pet emotions dataset, presenting the distribution of class labels using a bar plot.

### Data Visualization
Displaying sample images from each class of the pet emotions dataset along with their labels.

### Summary Statistics
Calculating summary statistics for each class (Angry, Sad, Happy, Others) to comprehend the distribution of emotions in the dataset.

## Dataset Task - Group Collaboration
Collection and basic operations on the "Common Objects-Within University" dataset.

## Lab 03: Enhancement and Fusion in Medical Image

### Improving and Analyzing Medical Image Quality
Tasks include loading and displaying X-ray images, contrast enhancement, color mapping, color balance, color filtering, logarithmic and power-law transformations.

### Enhancing Multi-Modal Medical Image Fusion
Tasks include loading X-ray and MRI images, histogram equalization, color mapping, multi-modal weighted fusion, logarithmic and power-law transformations, and comparative analysis.

### Real-Time Video Enhancement and Analysis
Capturing live video, applying various image enhancement operations in real-time, and displaying the original and enhanced video frames.

## Lab 04: Image Filtering and Fourier Transformations

### Linear Filtering
Implementing Gaussian blur, Sobel edge detection, image sharpening, and mean filter for noise reduction.

### Non-Linear Filtering
Developing median filter, max filter (dilation), min filter (erosion), bilateral filter, and adaptive median filter.

### Fourier Transformations
Calculating 1D and 2D Fourier Transforms, implementing high-pass filter, and performing image compression using Fourier Transformation.

### Hybrid Images
Creating hybrid images from two input images with different spatial frequencies, experimenting with filter combinations, and analyzing trade-offs.

## Lab 05: Medical Image Analysis and Feature Detection

### Medical Image Analysis for Tumor Detection
Discussing the application of edge detection as a feature extraction technique for tumor detection and proposing an additional feature extraction technique.

### Harris Corner Detection
Implementing the Harris Corner Detection algorithm, detecting corners in an image, and experimenting with different threshold values.

### Corner Detection in Real-time Video
Implementing corner detection in real-time using the Harris or Shi-Tomasi method on video frames.

### Corner Detection for Image Stitching
Implementing Harris or Shi-Tomasi Corner Detection for image stitching by detecting corners in multiple images.

### Feature Detection and Matching using ORB Detector
Utilizing ORB (Oriented FAST and Rotated BRIEF) detector and descriptor for feature detection and matching between two images.

## Lab 06: Image Segmentation

### Thresholding-Based Segmentation
Performing thresholding-based segmentation on a medical X-ray image to isolate a bone fracture.

### Region Growing Intensity-Based Segmentation
Performing region growing-based segmentation on a microscopic image of cells to identify and separate a specific cell.

### Watershed Segmentation
Using watershed segmentation to separate and count individual coins in an image of overlapping coins.

### Cluster-Based Segmentation
Performing cluster-based segmentation on an image of colorful flowers to separate different types of flowers based on color.

## Lab 07: Advanced Computer Vision Applications

### Computer Screen Detection
Implementing screen detection in a computer lab using the Hough Line Transformation to identify boundaries of computer screens.

### Asset Tracking Using SIFT
Implementing asset tracking in a computer lab using the Scale-Invariant Feature Transform (SIFT) to recognize and identify individual computer systems and components.

### Anomaly Detection Using Wavelet Transformation
Developing a system for real-time anomaly detection in sensor data using the wavelet transformation.

### Object Recognition (Using Video)
Implementing object recognition using the SIFT algorithm on a set of test images.

### Panoramic Image Stitching
Creating a panoramic image by stitching multiple overlapping images together using the SIFT algorithm.

### Lane Detection
Implementing lane detection for an autonomous vehicle project using the Hough Line Transformation.

### Coins Detection and Counting
Implementing coin detection and counting using the Hough Circle Transformation.

### Smart Security System
Implementing boundary detection in a security system to detect unauthorized objects in a predefined zone in a real-time video stream.

## Lab 08: Deep Learning for Computer Vision

### Gender Classification
Developing a CNN model for gender classification using a dataset of human faces labeled with gender information.

### Animal Facial Expression Recognition
Creating a CNN-based model for recognizing facial expressions in images of animals.

### Age Estimation
Building a system that estimates the age of a person in a video using a CNN-based architecture.

### Hand Gesture Recognition
Implementing a system for real-time hand gesture recognition using CNN models.

## Lab 09: Image Classification with Pre-trained Models

### Image Classification using EfficientNet and ResNet50
Performing image classification using pre-trained models, EfficientNet and ResNet50, on a chosen dataset.

## Lab 10: Object Detection with YOLO and R-CNN

### Lab 10-1: YOLO Object Detection
Using YOLO (You Only Look Once) to detect home assets using a live web camera.

### Lab 10-II: Object Detection Using R-CNN
Implementing object detection using Regional Convolutional Neural Networks (R-CNN) on a dataset of your choice.

## Lab 11: Image Classification with Vision Transformers

### Vision Transformer-based Image Classification
Classifying images using Vision Transformers on a dataset of your choice.
