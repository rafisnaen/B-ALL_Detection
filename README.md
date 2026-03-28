# B-ALL Detection using Convolutional Neural Networks (CNN)

This repository contains the CNN implementation for the multi-class classification of B-Cell Acute Lymphoblastic Leukemia (B-ALL) subtypes. [cite_start]This work is part of the research paper: **"Comparison between CNN, YOLOv11, and vision transformers deep learning architecture for B-ALL detection"** presented at the 10th International Conference on Computer Science and Computational Intelligence (ICCSCI) 2025.

[cite_start]While the paper presents a rigorous benchmark across three deep learning architectures (CNN, YOLOv11, ViT), **this specific repository focuses exclusively on the CNN architecture**, which achieved the highest overall performance in the study.

## 📊 Performance Highlights
Our custom CNN model demonstrated superior classification performance and stability compared to YOLOv11 and ViT, yielding the following results on the test dataset:
- **Test Accuracy**: 98.00%
- **Precision**: 0.9769
- **Recall**: 0.9755
- **F1-Score**: 0.9757
- **Macro AUC**: 0.9974
- **Categorical Cross-Entropy Loss**: 0.1009

## 🗂️ Dataset
The dataset utilized is a B-ALL white blood cell dataset obtained from the bone marrow laboratory of Taleqani Hospital, comprising 3,242 images. It is categorized into four classes:
1. **Benign** (512 images)
2. **Early Pre-B** (975 images)
3. **Pre-B** (955 images)
4. **Pro-B** (796 images)

The data is split into 70% Training, 20% Validation, and 10% Testing subsets.
> *Note: You can acquire the public dataset from Kaggle as referenced in the paper's Data Availability section.*

## 🏗️ Repository Structure
- `config.py`: Contains directory path configurations for the Train, Validation, and Test datasets.
- `CNN_Model.py`: The core script for data augmentation (rotation, shift, shear, zoom, flip), defining the CNN architecture, and training the model. The model is saved as `cnn_leukemia.h5`.
- `evaluate.py`: Script to evaluate the trained model on unseen test data. It outputs Accuracy, Confusion Matrix, Classification Report, Precision, Recall, F1-Score, AUC (Macro & Weighted), Loss, and generates ROC curves.
- `Predict.py`: A standalone inference script to load a single image, predict its leukemia subtype, and display the confidence percentage.
