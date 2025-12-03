import pandas as pd

train = pd.read_csv("/home/user4/Kaggle_Project/ck_train.csv")
test  = pd.read_csv("/home/user4/Kaggle_Project/ck_test.csv")   # and val if you made it

# add a split column if you want to remember where each row came from
train["split"] = "train"
test["split"]  = "test"

full = pd.concat([train, test], ignore_index=True)
full.to_csv("/home/user4/Kaggle_Project/ck_full.csv", index=False)
print("Saved /home/user4/Kaggle_Project/ck_full.csv with shape:", full.shape)
