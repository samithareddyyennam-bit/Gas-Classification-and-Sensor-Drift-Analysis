
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "outputs" / "gas_sensor_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)

# Load original dataset
data = pd.read_csv(DATA_PATH)

features = [f"feature_{i}" for i in range(1, 129)]

# Split by batches
train = data[data["batch"] <= 7]
validation = data[data["batch"].isin([8, 9])]

X_train = train[features]
X_val = validation[features]

# Encode labels using existing encoder
label_encoder = joblib.load(
    MODEL_DIR / "label_encoder.pkl"
)

y_train = label_encoder.transform(train["gas_name"])
y_val = label_encoder.transform(validation["gas_name"])

# Fit scaler ONLY on training batches 1-7
scaler = MinMaxScaler(feature_range=(-1, 1))

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define candidate models
models = {
    "Logistic Regression C=0.1": LogisticRegression(
        C=0.1,
        max_iter=3000,
        class_weight="balanced"
    ),

    "Logistic Regression C=1": LogisticRegression(
        C=1.0,
        max_iter=3000,
        class_weight="balanced"
    ),

    "Logistic Regression C=10": LogisticRegression(
        C=10,
        max_iter=3000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
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

results = []
best_model = None
best_name = None
best_f1 = -1

print("\n========== MODEL TUNING STARTED ==========")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_val_scaled)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    results.append({
        "Model": name,
        "Validation Accuracy": accuracy,
        "Macro F1": macro_f1
    })

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model = model
        best_name = name

# Display comparison
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

print("\n========== VALIDATION COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "validation_comparison.csv",
    index=False
)

# Save best validation model and its scaler
joblib.dump(
    best_model,
    MODEL_DIR / "best_validation_model.pkl"
)

joblib.dump(
    scaler,
    MODEL_DIR / "validation_scaler.pkl"
)

print("\nBest model:", best_name)
print("Best validation macro F1:", round(best_f1, 4))

print("\nSaved:")
print("outputs/validation_comparison.csv")
print("models/best_validation_model.pkl")
print("models/validation_scaler.pkl")

print("\n========== MODEL TUNING COMPLETED ==========")
