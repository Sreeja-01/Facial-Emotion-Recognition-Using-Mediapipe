import os
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = "/home/user4/Kaggle_Project/ckextended.csv"

# Read CSV
df = pd.read_csv(CSV_PATH)

# Column names from your description
EMOTION_COL = "emotion"
PIXELS_COL = "pixels"

# 80% train, 20% test (stratified by emotion labels)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df[EMOTION_COL],
    random_state=42,
)

print("Train size:", len(train_df))
print("Test size:", len(test_df))

# Save to new CSVs
train_df.to_csv("/home/user4/Kaggle_Project/ck_train.csv", index=False)
test_df.to_csv("/home/user4/Kaggle_Project/ck_test.csv", index=False)
