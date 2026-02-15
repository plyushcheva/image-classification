This repository contains two machine learning projects focused on classification tasks using image-based tabular data. 

## 1. Dimensionality Reduction & Binary Classification

### Goal

The objective of this project is to build a binary classification system for high-dimensional image data. It specifically explores how dimensionality reduction affects model accuracy and computational efficiency.

### Main Steps

* **Data Preparation:** Loading image data represented as pixel vectors and performing a train/validation/test split.
* **Feature Engineering:** Applying various normalization and scaling techniques including `MinMaxScaler`, `StandardScaler`, and `Binarizer`.
* **Initial Modeling:** Training baseline classifiers on the full feature set to establish benchmark performance.
* **Dimensionality Reduction:** Implementing **Principal Component Analysis (PCA)** and **Locally Linear Embedding (LLE)** to reduce the feature space.
* **Hyperparameter Tuning:** Using grid searches to optimize parameters (such as `C` for SVM or `var_smoothing` for Naive Bayes) within the reduced feature spaces.
* **Visualization:** Using the **LDA** model to generate and visualize synthetic data samples based on the learned class means and covariances.
* **Final Evaluation:** Comparing models based on validation accuracy to select the optimal configuration for final testing.

### Models Used

* **Support Vector Classifier (SVC)** with Linear and RBF kernels.
* **Naive Bayes** (Gaussian and Multinomial variants).
* **Linear Discriminant Analysis (LDA)**.




## 2. Image Classification with Neural Networks

### Goal

This project focuses on the application of Deep Learning to image classification. The goal is to design, train, and optimize neural networks, comparing standard dense architectures with spatial-aware convolutional layers.

### Main Steps

* **Neural Network Preprocessing:** Reshaping input data into appropriate tensors and normalizing pixel values for gradient-based optimization.
* **FNN Development:** Building a multi-layer **Feed-forward Neural Network** with hidden layers and ReLU activations.
* **Model Optimization:**
* Testing multiple optimizers such as **Adam** and **SGD**.
* Implementing **Dropout** layers to prevent overfitting.
* Using **Learning Rate Schedulers** and **Early Stopping** to ensure robust convergence.


* **CNN Implementation:** Developing a **Convolutional Neural Network** using 2D convolutions, max-pooling, and dropout to capture local patterns like edges and textures.
* **Performance Benchmarking:** Comparing the FNN and CNN architectures to evaluate the performance gains provided by convolutional layers.
* **Final Selection:** Saving the "best model" state during training and exporting final predictions to `results.csv`.

### Models Used

* **Feed-forward Neural Network (FNN)**.
* **Convolutional Neural Network (CNN)**.
