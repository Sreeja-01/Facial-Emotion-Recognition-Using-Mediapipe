📘 EmotionMesh — Facial Emotion Recognition Using Mediapipe & Machine Learning

This project implements facial emotion recognition using the CK+ (Cohn–Kanade Extended) dataset and Mediapipe Face Mesh.
Instead of raw pixel values, the system uses 478 facial landmarks (x,y,z) as geometric features (total 1434 features/sample) for emotion classification using SVM and Random Forest models.

🧠 Problem Statement

The goal is to classify facial expressions into 8 emotion categories using landmark-based features extracted from the CK+ dataset.
Mediapipe Face Mesh provides dense 3D face landmarks, enabling better recognition with limited training images.

🔧 Pipeline Overview
Dataset → Preprocessing → Mediapipe Face Mesh → Feature Extraction (N × 1434)
        → Train/Test Split → ML Models (SVM, RF) → Evaluation

🎯 Feature Extraction

Each image processed using Mediapipe Face Mesh (478 landmarks)

Each landmark has (x, y, z) → 478 × 3 = 1434 features

Features stored in .npy and .csv formats

Generated Files

ck_train_mediapipe_feats.npy

ck_test_mediapipe_feats.npy

ck_train_labels.npy

ck_full.csv

ck_train_mediapipe_feats.csv

ck_test_mediapipe_feats.csv

Dataset Dimensions

Training: 736 samples × 1434 features

Testing: 184 samples × 1434 features

Batch Size: 64

✂️ Train–Test Split

A stratified split ensures proportional class distribution:

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df[EMOTION_COL],
    random_state=42,
)

🧪 Model Training & Results
🔹 1. SVM Classifier (Multiple Kernels)
SVM – RBF Kernel

Accuracy: 79.34%

Performs well for majority classes

Poor for minority classes due to dataset imbalance

SVM – Sigmoid Kernel

Accuracy: 64.67%

Weak performance, fails on most minority classes

SVM – Polynomial Kernel (BEST MODEL)

Accuracy: ⭐ 85.32%

Captures nonlinear relations in facial geometry

Best overall precision and recall

SVM – Linear Kernel

Accuracy: 81.52%

Good performance but less flexible than Polynomial

SVM – Precomputed Kernel

Accuracy: 64.67%

Weak classification on smaller classes

🔹 2. Random Forest Classifier

Accuracy: 80.97%

Works well for majority classes

Lower recall for rare emotions (0,2,4,7)

Ensemble trees improve robustness

📊 Model Comparison Summary
Model	Accuracy
SVM – RBF	79.34%
SVM – Sigmoid	64.67%
SVM – Polynomial	⭐ 85.32% (Best)
SVM – Linear	81.52%
Random Forest	80.97%
📈 Why Do Some Classes Have 0 Precision/Recall?

CK+ dataset is imbalanced

Some classes (Fear, Sad, Disgust, Contempt) have very few samples

Models bias toward majority class (Neutral/Happy)

Solution: SMOTE, class weights, or deep learning in future versions

🚀 Tech Stack

Python

Mediapipe

NumPy, Pandas

Scikit-learn

Matplotlib

🔮 Future Improvements

Apply class-weighted training or SMOTE

Use deep learning: CNN + landmark fusion

Real-time emotion detection using webcam (OpenCV)

Deploy with Streamlit/Flask

🎉 Conclusion

This project demonstrates that Mediapipe 3D facial landmarks + ML models can classify emotions with high accuracy.
The Polynomial SVM delivers the best performance with 85.32% accuracy, showing strong potential for lightweight and real-time FER systems.
