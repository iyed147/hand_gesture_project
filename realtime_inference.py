import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib

MODEL_PATH = "best_gesture_model.joblib"
TASK_MODEL_PATH = "hand_landmarker.task"

def create_hand_landmarker():
    base_options = python.BaseOptions(model_asset_path=TASK_MODEL_PATH)

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
    features = []

    for lm in landmarks:
        features.extend([lm.x, lm.y])

    return features


def main():

    # Chargement modèle
    model = joblib.load(MODEL_PATH)
    print("Modele charge depuis", MODEL_PATH)

    # Détecteur MediaPipe
    detector = create_hand_landmarker()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur: impossible d'ouvrir la camera.")
        return

    current_gesture = "Unknown"

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Erreur lecture frame.")
            break

        # Correction effet miroir
        frame = cv2.flip(frame, 1)

        # BGR -> RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        # Détection main
        detection_result = detector.detect(mp_image)

        image_bgr = frame.copy()

        h, w, _ = image_bgr.shape

        if detection_result.hand_landmarks:

            landmarks = detection_result.hand_landmarks[0]

            # Dessin landmarks
            for lm in landmarks:

                x_px = int(lm.x * w)
                y_px = int(lm.y * h)

                cv2.circle(
                    image_bgr,
                    (x_px, y_px),
                    3,
                    (0, 255, 255),
                    -1
                )

            # Features
            features = extract_landmarks_normalized(landmarks)

            features_np = np.array(features).reshape(1, -1)

            # Prediction
            predicted_label = model.predict(features_np)[0]

            current_gesture = predicted_label

        # Texte affichage
        cv2.putText(
            image_bgr,
            f"Gesture: {current_gesture}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )

        cv2.imshow(
            "Real-time Gesture Recognition",
            image_bgr
        )

        # Quitter avec q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()