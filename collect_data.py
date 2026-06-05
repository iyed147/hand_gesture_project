import cv2
import numpy as np
import pandas as pd
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==== CONFIG ====
GESTURE_LABEL = "PERFECT"      # <-- change pour STOP, GOOD..
OUTPUT_CSV = "gestures_dataset.csv"
N_SAMPLES_PER_GESTURE = 200  # nombre de frames à collecter

# ==== MODELE HAND LANDMARKER ====
# Le modèle officiel MediaPipe pour les mains (format .task)
MODEL_PATH = "hand_landmarker.task"

def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        return
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    print("Téléchargement du modèle MediaPipe HandLandmarker...")
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Modèle téléchargé.")

def create_hand_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

def extract_landmarks_normalized(landmarks):
    """
    landmarks : liste de 21 points NormalizedLandmark (x,y,z)
    Retourne [x1, y1, ..., x21, y21]
    """
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y])  # coords normalisées [0,1]
    return features

def main():
    download_model_if_needed()
    detector = create_hand_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: impossible d'ouvrir la caméra.")
        return

    all_rows = []
    sample_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur: frame non lue.")
            break

        # OpenCV: BGR -> RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Création de l'objet Image MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Détection des mains
        detection_result = detector.detect(mp_image)

        image_bgr = frame.copy()
        h, w, _ = image_bgr.shape

        # Affichage du label courant
        cv2.putText(
            image_bgr,
            f"Label: {GESTURE_LABEL}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        if detection_result.hand_landmarks:
            # On ne prend que la première main (num_hands=1)
            landmarks = detection_result.hand_landmarks[0]

            # Dessin simple des points pour visualiser
            for lm in landmarks:
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                cv2.circle(image_bgr, (x_px, y_px), 3, (0, 255, 255), -1)

            # Extraction des features
            features = extract_landmarks_normalized(landmarks)
            row = features + [GESTURE_LABEL]
            all_rows.append(row)
            sample_count += 1

            cv2.putText(
                image_bgr,
                f"Samples: {sample_count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        cv2.imshow("Collect Data - q pour quitter", image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if sample_count >= N_SAMPLES_PER_GESTURE:
            print(f"{N_SAMPLES_PER_GESTURE} samples collectes pour {GESTURE_LABEL}.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Construction du DataFrame
    feature_cols = []
    for i in range(21):
        feature_cols.append(f"x{i+1}")
        feature_cols.append(f"y{i+1}")
    feature_cols.append("label")

    df_new = pd.DataFrame(all_rows, columns=feature_cols)

    # Si le CSV existe, on concatène; sinon on crée
    if os.path.exists(OUTPUT_CSV):
        df_old = pd.read_csv(OUTPUT_CSV)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset sauvegarde dans {OUTPUT_CSV}.")

if __name__ == "__main__":
    main()
