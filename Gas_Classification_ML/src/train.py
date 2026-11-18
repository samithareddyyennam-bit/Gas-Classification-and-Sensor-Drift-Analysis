
import numpy as np
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "outputs" / "processed_data.npz"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Load preprocessed data
data = np.load(DATA_PATH)

X_train = data["X_train"]
y_train = data["y_train"]

# Define ML algorithms
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced"
    )
}

print("\n========== MODEL TRAINING STARTED ==========")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(model, MODEL_DIR / filename)

    # Training accuracy (not test accuracy)
    train_accuracy = model.score(X_train, y_train)

    print(f"{name} training completed!")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Saved: models/{filename}")

print("\n========== ALL MODELS TRAINED ==========")
