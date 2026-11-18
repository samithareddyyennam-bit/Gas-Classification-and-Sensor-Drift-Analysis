
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Load dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "outputs" / "gas_sensor_dataset.csv"

PLOT_DIR = PROJECT_DIR / "outputs" / "eda_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

print("Creating EDA visualizations...")


# 2. Gas class distribution
plt.figure(figsize=(10, 6))

sns.countplot(
    data=data,
    x="gas_name",
    order=data["gas_name"].value_counts().index
)

plt.title("Gas Class Distribution")
plt.xlabel("Gas Type")
plt.ylabel("Number of Measurements")
plt.xticks(rotation=25)

plt.tight_layout()
plt.savefig(PLOT_DIR / "gas_distribution.png")
plt.close()


# 3. Batch distribution
plt.figure(figsize=(10, 6))

sns.countplot(
    data=data,
    x="batch",
    color="steelblue"
)

plt.title("Measurements Across Different Batches")
plt.xlabel("Batch Number")
plt.ylabel("Number of Measurements")

plt.tight_layout()
plt.savefig(PLOT_DIR / "batch_distribution.png")
plt.close()


# 4. Feature value distribution
plt.figure(figsize=(10, 6))

sns.histplot(
    data=data,
    x="feature_1",
    bins=50,
    kde=True
)

plt.title("Distribution of Sensor Feature 1")
plt.xlabel("Feature 1 Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_distribution.png")
plt.close()


# 5. Compare feature values across gas classes
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=data,
    x="gas_name",
    y="feature_1"
)

plt.title("Feature 1 Comparison Across Gas Classes")
plt.xlabel("Gas Type")
plt.ylabel("Feature 1 Value")
plt.xticks(rotation=25)

plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_by_gas.png")
plt.close()


# 6. Correlation heatmap
# Select first 20 features for readable visualization
features = [
    f"feature_{i}" for i in range(1, 21)
]

correlation = data[features].corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    annot=False
)

plt.title("Correlation Heatmap of First 20 Sensor Features")

plt.tight_layout()
plt.savefig(PLOT_DIR / "correlation_heatmap.png")
plt.close()


print("\nAll visualizations created successfully!")

print("\nSaved plots inside:")
print(PLOT_DIR)

print("\nGenerated files:")
for file in PLOT_DIR.glob("*.png"):
    print(file.name)
