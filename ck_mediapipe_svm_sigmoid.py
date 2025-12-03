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

    # 2. Define SVM with SIGMOID kernel
    clf = SVC(
        kernel="sigmoid",
        C=10,          # you can tune this
        gamma="scale", # or try a float, e.g. 0.001
        coef0=0.0,     # bias term in sigmoid, you can also tune
    )

    # 3. Train on TRAIN set only
    clf.fit(X_train, y_train)

    # 4. Evaluate on TEST set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy (sigmoid kernel):", acc)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
