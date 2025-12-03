import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

CSV_PATH = "/home/user4/Kaggle_Project/ck_train.csv"
IMG_SIZE = 48
BATCH_SIZE = 64  # batch size for reading CSV

OUT_FEATS = "/home/user4/Kaggle_Project/ck_train_mediapipe_feats.npy"
OUT_LABELS = "/home/user4/Kaggle_Project/ck_train_labels.npy"

mp_face_mesh = mp.solutions.face_mesh

def pixels_to_image(pixels_str):
    arr = np.fromstring(pixels_str, sep=" ", dtype=np.uint8)
    img = arr.reshape(IMG_SIZE, IMG_SIZE)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_bgr

def draw_landmarks(image_bgr, landmarks):
    h, w, _ = image_bgr.shape
    out = image_bgr.copy()
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(out, (x, y), 1, (0, 255, 0), -1)
    return out

def build_features_text(landmarks, emotion_label):
    lines = []
    lines.append(f"Emotion: {emotion_label}")
    lines.append(f"Num landmarks: {len(landmarks)}")
    lines.append("Index   x        y        z")
    for idx, lm in enumerate(landmarks):
        lines.append(f"{idx:3d}   {lm.x:.4f}  {lm.y:.4f}  {lm.z:.4f}")
    return "\n".join(lines)

def put_multiline_text(img, text, x, y, line_height=18):
    for i, line in enumerate(text.split("\n")):
        cv2.putText(
            img,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

def main():
    # lists to collect feature vectors and labels for ALL batches
    all_feats = []
    all_labels = []

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # read CSV in batches
    reader = pd.read_csv(CSV_PATH, chunksize=BATCH_SIZE)

    for chunk_idx, df_chunk in enumerate(reader):
        print(f"Processing chunk {chunk_idx}")

        # index inside this chunk
        idx = 0
        while 0 <= idx < len(df_chunk):
            row = df_chunk.iloc[idx]
            emotion = int(row["emotion"])
            pixels_str = row["pixels"]

            img_left = pixels_to_image(pixels_str)
            img_right = img_left.copy()

            rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            feature_text = f"Emotion: {emotion}\nNo face detected."

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                img_right = draw_landmarks(img_right, face_landmarks)
                feature_text = build_features_text(face_landmarks, emotion)

                coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks],
                                  dtype=np.float32)
                feat_vec = coords.flatten()
                all_feats.append(feat_vec)
                all_labels.append(emotion)

            vis = np.hstack([img_left, img_right])

            scale = 6
            vis = cv2.resize(
                vis,
                (vis.shape[1] * scale, vis.shape[0] * scale),
                interpolation=cv2.INTER_NEAREST,
            )

            bottom_height = 1000
            vis_with_text = np.zeros(
                (vis.shape[0] + bottom_height, vis.shape[1], 3), dtype=np.uint8
            )
            vis_with_text[:vis.shape[0], :, :] = vis

            cv2.rectangle(
                vis_with_text,
                (0, vis.shape[0]),
                (vis.shape[1], vis.shape[0] + bottom_height),
                (30, 30, 30),
                -1,
            )

            put_multiline_text(
                vis_with_text,
                feature_text,
                10,
                vis.shape[0] + 25,
                line_height=18,
            )

            cv2.imshow(
                "CK+ MediaPipe Viewer (left: input, right: landmarks)",
                vis_with_text,
            )
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                # stop everything immediately
                face_mesh.close()
                cv2.destroyAllWindows()
                if all_feats:
                    all_feats_arr = np.stack(all_feats, axis=0)
                    all_labels_arr = np.array(all_labels, dtype=np.int64)
                    np.save(OUT_FEATS, all_feats_arr)
                    np.save(OUT_LABELS, all_labels_arr)
                    print("Saved feature matrix to:", OUT_FEATS)
                    print("Saved labels to:", OUT_LABELS)
                    print("Feature matrix shape:", all_feats_arr.shape)
                    print("Labels shape:", all_labels_arr.shape)
                else:
                    print("No faces detected, nothing saved.")
                return
            elif key == ord("n"):
                idx += 1
            elif key == ord("b"):
                idx -= 1
            else:
                idx += 1

    face_mesh.close()
    cv2.destroyAllWindows()

    if all_feats:
        all_feats_arr = np.stack(all_feats, axis=0)
        all_labels_arr = np.array(all_labels, dtype=np.int64)

        np.save(OUT_FEATS, all_feats_arr)
        np.save(OUT_LABELS, all_labels_arr)
        print("Saved feature matrix to:", OUT_FEATS)
        print("Saved labels to:", OUT_LABELS)
        print("Feature matrix shape:", all_feats_arr.shape)
        print("Labels shape:", all_labels_arr.shape)
    else:
        print("No faces detected, nothing saved.")

if __name__ == "__main__":
    main()
