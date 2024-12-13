# Pneumonia Detection on X-Ray Images of Chest

## Project Overview

This project aims to detect pneumonia in chest X-ray images using a deep learning-based Convolutional Neural Network (CNN). The goal is to create a prescreening tool for healthcare providers, particularly in under-resourced areas, to improve diagnostic accuracy and speed. The solution identifies pneumonia and highlights affected regions, offering an explainable AI tool for radiologists.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Methods and Approach](#methods-and-approach)
   - [Data Preparation](#data-preparation)
   - [Preprocessing](#preprocessing)
   - [Data Augmentation and Generator Design](#data-augmentation-and-generator-design)
   - [Model Architecture](#model-architecture)
   - [Training and Validation](#training-and-validation)
   - [Testing and Evaluation](#testing-and-evaluation)
   - [Visualization and Analysis](#visualization-and-analysis)
3. [Results and Analysis](#results-and-analysis)
4. [Conclusion and Reflection](#conclusion-and-reflection)
5. [References](#references)

---

## Introduction

Pneumonia remains a leading cause of death worldwide, with over 50,000 fatalities in the U.S. in 2015. Accurate diagnosis often requires expert review of chest radiographs (CXRs), which is time-intensive and prone to errors. This project leverages CNN to detect pneumonia from CXRs, aiming to assist radiologists and bridge the gap between advanced diagnostics and accessibility challenges.

---

## Methods and Approach

### Data Preparation
- Dataset: 30,000 CXR images in DICOM format with bounding box annotations.
- Splitting: 80% for training and 20% for testing.
- Organization: Images separated into directories for efficient data loading.

### Preprocessing
- Resize images to 224×224 pixels.
- Normalize pixel values to [0,1].
- Combine metadata at the patient level for multiple pneumonia regions.
- Binary labels: `0` for no pneumonia, `1` for pneumonia.

### Data Augmentation and Generator Design
- Custom data generator using `keras.utils.Sequence` for dynamic, memory-efficient loading.
- Uniform preprocessing transformations applied across the dataset.

### Model Architecture
- **CNN Enhancements:**
  - L2 Regularization and dropout layers to reduce overfitting.
  - Batch normalization to stabilize training.
  - ReLU activation for hidden layers and sigmoid activation for the output layer.
- **Optimizer:** Adam
- **Loss Function:** Binary cross-entropy

### Training and Validation
- **Callbacks:**
  - ModelCheckpoint: Save the best-performing model.
  - EarlyStopping: Prevent overfitting by halting training early.
  - ReduceLROnPlateau: Adjust learning rate when validation loss plateaued.

### Testing and Evaluation
- Metrics: Confusion matrix, precision, recall, F1 score, and AUC-ROC.
- Threshold optimization to balance sensitivity and specificity.

### Visualization and Analysis
- Learning curves: Analyzed convergence and overfitting.
- Performance visualizations: Confusion matrix, ROC curve, and accuracy vs. threshold plots.

---

## Results and Analysis

- **Performance Metrics:**
  - Test accuracy: 80%
  - AUC-ROC score: 0.81
  - High specificity but lower sensitivity due to overlapping visual features.
- **Key Insights:**
  - Training and validation loss converged, indicating effective optimization.
  - The confusion matrix showed robust detection of non-pneumonia cases but room for improvement in sensitivity for pneumonia cases.

---

## Conclusion and Reflection

The project successfully developed a CNN model achieving a test accuracy of 80% and AUC-ROC of 0.81. While the model demonstrated strong performance in detecting non-pneumonia cases, challenges remain in improving sensitivity. Future directions include:
- Addressing class imbalance with augmentation and class weighting.
- Incorporating Grad-CAM for enhanced explainability.
- Expanding the dataset for greater diversity.

This project highlights the potential of AI in medical imaging for timely and accurate diagnoses, particularly in resource-limited settings.

---

## References

1. RSNA Pneumonia Detection Challenge Dataset: [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
2. Litjens, G., et al. (2017). Deep learning in medical image analysis. Medical Image Analysis, 42.
3. Rajpurkar, P., et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection. [arXiv:1711.05225](https://arxiv.org/abs/1711.05225)
4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations. International Journal of Computer Vision.
5. World Health Organization (WHO): [Pneumonia Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/pneumonia)
