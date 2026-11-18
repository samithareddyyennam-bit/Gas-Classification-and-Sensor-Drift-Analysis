
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_svmlight_file

# 1. Define project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "dataset" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Define gas names
gas_names = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene"
}

# 3. Find all batch files
batch_files = sorted(
    DATA_DIR.glob("batch*.dat"),
    key=lambda file: int(file.stem.replace("batch", ""))
)

if len(batch_files) != 10:
    raise FileNotFoundError(
        f"Expected 10 batch files, but found {len(batch_files)}. "
        "Check dataset/raw folder."
    )

all_data = []

# 4. Load each batch
for file in batch_files:

    print(f"Loading: {file.name}")

    X, y = load_svmlight_file(
        str(file),
        n_features=128,
        zero_based=False
    )

    # Convert sparse matrix to normal array
    X = X.toarray()

    # Create DataFrame
    df = pd.DataFrame(
        X,
        columns=[f"feature_{i}" for i in range(1, 129)]
    )

    # Add target information
    df["gas_id"] = y.astype(int)
    df["gas_name"] = df["gas_id"].map(gas_names)

    # Preserve batch information
    df["batch"] = int(
        file.stem.replace("batch", "")
    )

    all_data.append(df)

# 5. Combine all batches
data = pd.concat(all_data, ignore_index=True)

# 6. Save processed dataset
output_file = OUTPUT_DIR / "gas_sensor_dataset.csv"

data.to_csv(output_file, index=False)

# 7. Display dataset information
print("\n========== DATASET LOADING SUCCESS ==========")

print("\nDataset shape:")
print(data.shape)

print("\nFirst five rows:")
print(data.head())

print("\nGas class distribution:")
print(data["gas_name"].value_counts())

print("\nBatch distribution:")
print(data["batch"].value_counts().sort_index())

print("\nMissing values:")
print(data.isnull().sum().sum())

print("\nSaved dataset to:")
print(output_file)
