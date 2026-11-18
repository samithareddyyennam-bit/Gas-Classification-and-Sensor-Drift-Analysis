
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "outputs" / "gas_sensor_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("\n========== FINAL MODEL TRAINING ==========")

data = pd.read_csv(DATA_PATH)
features = [f"feature_{i}" for i in range(1, 129)]

# Training: batches 1–9
train = data[data["batch"] <= 9]

# Final testing: batch 10
test = data[data["batch"] == 10]

X_train = train[features]
X_test = test[features]

label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

y_train = label_encoder.transform(train["gas_name"])
y_test = label_encoder.transform(test["gas_name"])

# Fit scaler only on training data
scaler = MinMaxScaler(feature_range=(-1, 1))

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Best model selected during validation
final_model = LogisticRegression(
    C=10,
    max_iter=3000,
    class_weight="balanced"
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining final model...")
final_model.fit(X_train_scaled, y_train)

# Predictions
predictions = final_model.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="weighted")
recall = recall_score(y_test, predictions, average="weighted")
macro_f1 = f1_score(y_test, predictions, average="macro")
weighted_f1 = f1_score(y_test, predictions, average="weighted")

print("\n========== FINAL TEST RESULTS ==========")
print(f"Accuracy:           {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall:    {recall:.4f}")
print(f"Macro F1:           {macro_f1:.4f}")
print(f"Weighted F1:        {weighted_f1:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test,
    predictions,
    target_names=label_encoder.classes_
))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Final Model - Batch 10 Confusion Matrix")
plt.xlabel("Predicted Gas")
plt.ylabel("Actual Gas")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_confusion_matrix.png",
    dpi=300
)

plt.close()

# Save final model and scaler
joblib.dump(final_model, MODEL_DIR / "final_gas_classifier.pkl")
joblib.dump(scaler, MODEL_DIR / "final_scaler.pkl")

print("\nSaved final model and scaler.")
print("Saved final confusion matrix.")

print("\n========== FINAL TRAINING COMPLETED ==========")
