import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TRAIN_CSV = "/home/user4/Kaggle_Project/ck_train_mediapipe_feats.csv"
TEST_CSV  = "/home/user4/Kaggle_Project/ck_test_mediapipe_feats.csv"

def load_xy(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["emotion"]).values  # features
    y = df["emotion"].values                # labels
    return X, y

def main():
    # 1. Load train / test
    X_train, y_train = load_xy(TRAIN_CSV)
    X_test, y_test   = load_xy(TEST_CSV)

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # 2. Polynomial kernel SVM
    clf = SVC(
        kernel="poly",
        degree=3,        # polynomial degree
        C=10,            # regularization
        gamma="scale",   # kernel coefficient
        coef0=1.0        # bias term inside the polynomial
    )

    # 3. Train on TRAIN set
    clf.fit(X_train, y_train)

    # 4. Evaluate on TEST set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy (poly kernel):", acc)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
