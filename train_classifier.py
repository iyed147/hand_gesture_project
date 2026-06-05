import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

DATA_CSV = "gestures_dataset.csv"
MODEL_PATH = "best_gesture_model.joblib"

def main():
    # Chargement du dataset
    df = pd.read_csv(DATA_CSV)

    # Séparation X / y
    feature_cols = [c for c in df.columns if c.startswith("x") or c.startswith("y")]
    X = df[feature_cols].values
    y = df["label"].values

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Définition des 4 modèles
    models = {
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
    }

    best_name = None
    best_acc = 0.0
    best_model = None

    # Entraînement & évaluation
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    print(f"\nMeilleur modele: {best_name} avec accuracy = {best_acc:.4f}")

    # Sauvegarde du meilleur modèle
    joblib.dump(best_model, MODEL_PATH)
    print(f"Modele sauvegarde dans {MODEL_PATH}")

if __name__ == "__main__":
    main()
