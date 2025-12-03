import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

CSV_PATH = "/home/user4/Kaggle_Project/ck_full.csv"
IMG_SIZE = 48

OUT_NPY_FEATS = "/home/user4/Kaggle_Project/ck_full_mediapipe_feats.npy"
OUT_NPY_LABELS = "/home/user4/Kaggle_Project/ck_full_labels.npy"
OUT_CSV = "/home/user4/Kaggle_Project/ck_full_mediapipe_feats.csv"

mp_face_mesh = mp.solutions.face_mesh

def pixels_to_image(pixels_str):
    arr = np.fromstring(pixels_str, sep=" ", dtype=np.uint8)
    img = arr.reshape(IMG_SIZE, IMG_SIZE)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_bgr

def main():
    df = pd.read_csv(CSV_PATH)

    all_feats = []
    all_labels = []

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    for idx, row in df.iterrows():
        emotion = int(row["emotion"])
        pixels_str = row["pixels"]

        img = pixels_to_image(pixels_str)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        face_landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks],
                          dtype=np.float32)
        feat_vec = coords.flatten()
        all_feats.append(feat_vec)
        all_labels.append(emotion)

    face_mesh.close()

    if not all_feats:
        print("No faces detected, nothing saved.")
        return

    all_feats = np.stack(all_feats, axis=0)
    all_labels = np.array(all_labels, dtype=int)

    # optional NPY
    np.save(OUT_NPY_FEATS, all_feats)
    np.save(OUT_NPY_LABELS, all_labels)
    print("Saved NPY:", OUT_NPY_FEATS, OUT_NPY_LABELS)
    print("NPY shapes:", all_feats.shape, all_labels.shape)

    # CSV table
    num_landmarks = all_feats.shape[1] // 3
    cols = []
    for i in range(num_landmarks):
        cols.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])

    df_feats = pd.DataFrame(all_feats, columns=cols)
    df_feats.insert(0, "emotion", all_labels)

    # if you added "split" column in ck_full.csv and want to keep it:
    if "split" in df.columns:
        # WARNING: some rows may have been skipped (no face), so indices may not align perfectly.
        # Simple version: just drop 'split' from the feature CSV, or handle carefully.
        pass

    df_feats.to_csv(OUT_CSV, index=False)
    print("Saved CSV feature table to:", OUT_CSV)
    print("CSV shape:", df_feats.shape)

if __name__ == "__main__":
    main()
