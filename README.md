# CNN-FashionMNIST-Project-Pranto

# Project Overview
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify fashion items from the Fashion-MNIST dataset.
After training on a standard benchmark dataset, the model is evaluated on real-world images captured using a smartphone to analyze model generalization and domain shift.
The complete workflow is fully automated and reproducible using Google Colab and GitHub.

# Repository Structure
CNN-FashionMNIST-Project-Pranto/
│
├── 210102.ipynb                 # Complete Google Colab notebook
│
├── kaggle_data/                 # Fashion-MNIST dataset (CSV, Git LFS)
│   ├── fashion-mnist_train.csv
│   └── fashion-mnist_test.csv
│
├── dataset/                     # Real-world phone images
│   ├── bag1.jpg
│   ├── bag2.jpg
│   ├── sandal1.jpg
│   ├── shirt1.jpg
│   ├── shirt2.jpg
│   ├── sneaker1.jpg
│   ├── trouser1.jpg
│   └── trouser2.jpg
│
├── model/
│   └── 210102.pth               # Trained CNN model (state_dict)
│
├── results/
│   └── phone_predictions.png    # Real-world prediction visualization
│
└── README.md

# Dataset Description
Standard Dataset (Training & Testing)

# Fashion-MNIST

60,000 training images

10,000 test images

Image size: 28 × 28 grayscale

10 classes:

    T-shirt/top
    
    Trouser
    
    Pullover
    
    Dress
    
    Coat
    
    Sandal
    
    Shirt
    
    Sneaker
    
    Bag
    
    Ankle boot

The dataset is loaded automatically from CSV files stored in the repository using Git LFS.

Real-World Dataset (Phone Images)

Images captured using a smartphone

# Objects include:

Shirt

Trouser

Sneaker

Sandal

Bag

Images are resized and converted to grayscale to match Fashion-MNIST format

These images are not used for training, only for testing generalization.

CNN Model Architecture

# The CNN model consists of:

Convolutional Layer (Conv2D + ReLU)

MaxPooling Layer

Second Convolutional Layer

Fully Connected Layers

Output layer with 10 neurons (one per class)

Loss Function: CrossEntropyLoss
# Optimizer: Adam

The model is implemented using torch.nn.Module.

# Training & Evaluation

The model is trained for multiple epochs on Fashion-MNIST

Training and validation loss and accuracy plots are generated

Performance is evaluated on the standard test set

A confusion matrix is generated to analyze class-wise performance

The notebook executes all steps automatically when Run All is selected in Colab.

# Confusion Matrix (Fashion-MNIST Test Set)

The confusion matrix shows strong diagonal dominance, indicating good classification performance on the standard test dataset.
Most misclassifications occur between visually similar classes such as Shirt and T-shirt/top, which is expected for Fashion-MNIST.

#Real-World Phone Image Predictions

The trained CNN is evaluated on real-world phone images using the same preprocessing pipeline as the training data (grayscale conversion, resizing, normalization).

# Prediction Results

Important: The percentage shown below each image represents the model’s confidence (softmax probability) for the predicted class.
It should not be interpreted as real-world accuracy.

# Accuracy vs Confidence (Important Clarification)

The confidence values shown for real-world images are derived from the softmax output of the CNN and indicate how confident the model is in its prediction, not whether the prediction is correct.

Although confidence values are high, predictions on phone images are often biased toward the “Bag” class. This occurs due to domain shift between:

Fashion-MNIST’s low-resolution, silhouette-style images

Real photographs containing complex backgrounds, lighting, shadows, and textures

This behavior highlights the limitations of models trained on synthetic datasets when applied to real-world data.

# How to Run the Project

Open 210102.ipynb from GitHub

Click Open in Colab

Select Runtime → Run all

# The notebook will:

Clone the repository

Load datasets automatically

Train the CNN

Generate plots and confusion matrix

Predict real-world phone images with confidence scores

No manual file upload is required.

Key Learning Outcomes

Understanding CNN architecture using PyTorch

Image preprocessing for deep learning

Model evaluation using confusion matrices

Observing domain shift in real-world deployment

Using GitHub and Google Colab for reproducible ML workflows

# Author

Pranto Bala
Department of Computer Science & Engineering, Jashore University of Science and Technology

GitHub Repository: This repository

Google Colab Notebook: Open via GitHub
