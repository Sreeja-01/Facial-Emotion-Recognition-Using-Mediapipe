import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel  # helper function

TRAIN_CSV = "/home/user4/Kaggle_Project/ck_train_mediapipe_feats.csv"
TEST_CSV  = "/home/user4/Kaggle_Project/ck_test_mediapipe_feats.csv"

def load_xy(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["emotion"]).values  # features
    y = df["emotion"].values                # labels
    return X, y

def main():
    X_train, y_train = load_xy(TRAIN_CSV)
    X_test, y_test   = load_xy(TEST_CSV)

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # 1. Compute precomputed kernel matrices
    # Choose gamma (you can tune this)
    gamma = 0.001  # example; you can try other values

    # K_train: (n_train, n_train)
    K_train = rbf_kernel(X_train, X_train, gamma=gamma)

    # K_test: (n_test, n_train)
    K_test = rbf_kernel(X_test, X_train, gamma=gamma)

    # 2. Define SVM with precomputed kernel
    clf = SVC(
        kernel="precomputed",
        C=10,  # regularization parameter
    )

    # 3. Train on precomputed train kernel
    clf.fit(K_train, y_train)

    # 4. Predict using precomputed test kernel
    y_pred = clf.predict(K_test)

    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy (precomputed RBF kernel):", acc)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
