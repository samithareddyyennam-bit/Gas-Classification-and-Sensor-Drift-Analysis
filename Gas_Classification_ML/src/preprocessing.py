
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "outputs" / "gas_sensor_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
data = pd.read_csv(DATA_PATH)

# Separate features and target
feature_columns = [f"feature_{i}" for i in range(1, 129)]

X = data[feature_columns]
y = data["gas_name"]
batch = data["batch"]

# Split using batch numbers
train_mask = batch <= 9
test_mask = batch == 10

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

# Encode gas labels
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Scale features
scaler = MinMaxScaler(feature_range=(-1, 1))

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessing objects
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")

# Save processed datasets
np.savez(
    OUTPUT_DIR / "processed_data.npz",
    X_train=X_train_scaled,
    X_test=X_test_scaled,
    y_train=y_train_encoded,
    y_test=y_test_encoded
)

print("\n========== PREPROCESSING COMPLETED ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of features:", X_train.shape[1])

print("\nGas classes:")
print(label_encoder.classes_)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nScaled training range:")
print(X_train_scaled.min(), "to", X_train_scaled.max())

print("\nScaled testing range:")
print(X_test_scaled.min(), "to", X_test_scaled.max())

print("\nSaved files:")
print("models/scaler.pkl")
print("models/label_encoder.pkl")
print("outputs/processed_data.npz")
