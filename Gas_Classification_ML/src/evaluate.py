
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "outputs" / "processed_data.npz"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

PLOT_DIR = OUTPUT_DIR / "evaluation_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load test data
data = np.load(DATA_PATH)

X_test = data["X_test"]
y_test = data["y_test"]

# Load label encoder
label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
class_names = label_encoder.classes_

# Load trained models
models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "SVM": "svm.pkl"
}

results = []

print("\n========== MODEL EVALUATION STARTED ==========")

for name, filename in models.items():

    print(f"\nEvaluating {name}...")

    model = joblib.load(MODEL_DIR / filename)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{name} Test Accuracy: {accuracy:.4f}")

    # Detailed classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Gas")
    plt.ylabel("Actual Gas")
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_name = name.lower().replace(" ", "_")
    plt.savefig(PLOT_DIR / f"{plot_name}_confusion_matrix.png")
    plt.close()

    # Store metrics
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1 Score": report["weighted avg"]["f1-score"]
    })

# Compare all models
results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nSaved comparison:")
print("outputs/model_comparison.csv")

print("\nSaved confusion matrices:")
print(PLOT_DIR)

print("\n========== EVALUATION COMPLETED ==========")
