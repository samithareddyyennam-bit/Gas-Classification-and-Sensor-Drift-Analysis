
import pandas as pd
from pathlib import Path

# 1. Load the generated dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "outputs" / "gas_sensor_dataset.csv"

data = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

# 2. Dataset dimensions
print("\nDataset Shape:")
print(data.shape)

# 3. Column information
print("\nColumn Information:")
print(data.info())

# 4. Statistical summary
print("\nStatistical Summary:")
print(data.describe())

# 5. Missing values
print("\nMissing Values:")
print(data.isnull().sum().sum())

# 6. Duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())

# 7. Gas class counts
print("\nGas Class Distribution:")
print(data["gas_name"].value_counts())

# 8. Batch counts
print("\nBatch Distribution:")
print(data["batch"].value_counts().sort_index())

print("\n========== INSPECTION COMPLETED ==========")
print("\n========== ADVANCED EDA REPORT ==========")

# 1. Check feature skewness
features = [f"feature_{i}" for i in range(1, 129)]

skewness = data[features].skew().sort_values(
    key=abs, ascending=False
)

print("\nTop 10 Most Skewed Features:")
print(skewness.head(10))

# 2. Check extreme values
print("\nFeature Value Percentiles:")
print(data[features].quantile([0, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).iloc[:, :5])

# 3. Check gas classes across batches
print("\nGas Distribution Across Batches:")
print(pd.crosstab(data["batch"], data["gas_name"]))

# 4. Check highly correlated features
correlation = data[features].corr().abs()
correlation_pairs = correlation.where(
    ~pd.DataFrame(
        __import__("numpy").tril(
            __import__("numpy").ones(correlation.shape)
        ).astype(bool),
        index=correlation.index,
        columns=correlation.columns
    )
).stack().sort_values(ascending=False)

print("\nTop 10 Highly Correlated Feature Pairs:")
print(correlation_pairs.head(10))

print("\n========== ADVANCED EDA COMPLETED ==========")

print("\n========== VALIDATION BATCH CHECK ==========")

validation_data = data[data["batch"].isin([8, 9])]

print("\nGas distribution in validation batches:")
print(
    pd.crosstab(
        validation_data["batch"],
        validation_data["gas_name"]
    )
)

print("\nGas classes available in validation data:")
print(validation_data["gas_name"].nunique())

print("\nValidation samples:")
print(len(validation_data))

print("\n========== CHECK COMPLETED ==========")
