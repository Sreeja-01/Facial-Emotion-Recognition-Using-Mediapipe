import numpy as np
import pandas as pd
import cv2

from feat import Detector   # Py-Feat
from tqdm import tqdm       # for nice progress bar (optional)

CSV_TRAIN = "/home/user4/Kaggle_Project/ck_train.csv"
IMG_SIZE = 48
BATCH_SIZE = 64

OUT_FEATS = "/home/user4/Kaggle_Project/ck_train_pyfeat_feats.npy"
OUT_LABELS = "/home/user4/Kaggle_Project/ck_train_pyfeat_labels.npy"

def pixels_to_image(pixels_str):
    arr = np.fromstring(pixels_str, sep=" ", dtype=np.uint8)
    img = arr.reshape(IMG_SIZE, IMG_SIZE)           # 48x48
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Py-Feat expects color
    return img_bgr

def main():
    # Initialize Py-Feat detector (default models)
    detector = Detector()

    all_feats = []
    all_labels = []

    # Read CK+ CSV in batches
    reader = pd.read_csv(CSV_TRAIN, chunksize=BATCH_SIZE)

    for chunk_idx, df_chunk in enumerate(reader):
        print(f"Processing chunk {chunk_idx}")

        # Convert this chunk to list of images and labels
        imgs = []
        labels = []
        for _, row in df_chunk.iterrows():
            emotion = int(row["emotion"])
            pixels_str = row["pixels"]
            img = pixels_to_image(pixels_str)
            imgs.append(img)
            labels.append(emotion)

        if not imgs:
            continue

        # Run Py-Feat on this batch of images
        # data_type="image" and as_batch=True means list of images
        fex = detector.detect(
            imgs,
            data_type="image",
            as_batch=True,
        )

        # fex is a Fex dataframe: many columns (AU, emotions, landmarks, etc.)
        # Choose which columns to use as features.
        # Example: use all numeric feature columns except meta columns.
        # You can inspect fex.columns once to refine this.
        meta_cols = ["input", "frame", "face_id"]  # depending on version
        feature_cols = [c for c in fex.columns if c not in meta_cols]

        # Convert selected columns to numpy
        feats_chunk = fex[feature_cols].to_numpy(dtype=np.float32)

        # feats_chunk shape: (batch_size, D_pyfeat)
        all_feats.append(feats_chunk)
        all_labels.extend(labels)

    if not all_feats:
        print("No features extracted, nothing saved.")
        return

    # Concatenate all batches
    all_feats = np.vstack(all_feats)            # shape (N, D_pyfeat)
    all_labels = np.array(all_labels, int)      # shape (N,)

    np.save(OUT_FEATS, all_feats)
    np.save(OUT_LABELS, all_labels)

    print("Saved Py-Feat feature matrix to:", OUT_FEATS)
    print("Saved labels to:", OUT_LABELS)
    print("Feature matrix shape:", all_feats.shape)
    print("Labels shape:", all_labels.shape)

if __name__ == "__main__":
    main()
