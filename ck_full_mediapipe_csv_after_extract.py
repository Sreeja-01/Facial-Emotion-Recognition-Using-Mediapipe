import pandas as pd
from sklearn.model_selection import train_test_split

FULL_CSV = "/home/user4/Kaggle_Project/ck_full_mediapipe_feats.csv"
TRAIN_CSV = "/home/user4/Kaggle_Project/ck_train_mediapipe_feats.csv"
TEST_CSV  = "/home/user4/Kaggle_Project/ck_test_mediapipe_feats.csv"

df = pd.read_csv(FULL_CSV)

# X = all feature columns, y = emotion label
X = df.drop(columns=["emotion"])
y = df["emotion"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,          # 20% test, 80% train
    stratify=y,             # keep emotion proportions
    random_state=42,
)

# Recombine X and y so you keep same format
train_df = X_train.copy()
train_df["emotion"] = y_train

test_df = X_test.copy()
test_df["emotion"] = y_test

# Put emotion as first column (optional, like original)
cols = ["emotion"] + [c for c in train_df.columns if c != "emotion"]
train_df = train_df[cols]
test_df = test_df[cols]

train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

