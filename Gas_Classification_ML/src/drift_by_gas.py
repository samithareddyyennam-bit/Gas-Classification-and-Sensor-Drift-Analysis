
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "outputs" / "gas_sensor_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

data = pd.read_csv(DATA_PATH)

features = [f"feature_{i}" for i in range(1, 129)]
gas_classes = sorted(data["gas_name"].unique())

train = data[data["batch"] <= 9]
test = data[data["batch"] == 10]

results = []

print("\n========== GAS-WISE DRIFT VALIDATION ==========")

for gas in gas_classes:

    train_gas = train[train["gas_name"] == gas]
    test_gas = test[test["gas_name"] == gas]

    print(
        f"\n{gas}: Training={len(train_gas)}, "
        f"Testing={len(test_gas)}"
    )

    for feature in features:

        statistic, p_value = ks_2samp(
            train_gas[feature],
            test_gas[feature]
        )

        results.append({
            "Gas": gas,
            "Feature": feature,
            "KS_Statistic": statistic,
            "P_Value": p_value
        })

results_df = pd.DataFrame(results)

# Correct for multiple statistical tests
number_of_tests = len(results_df)
bonferroni_threshold = 0.05 / number_of_tests

results_df["Significant"] = (
    results_df["P_Value"] < bonferroni_threshold
)

results_df.to_csv(
    OUTPUT_DIR / "gas_wise_drift_results.csv",
    index=False
)

print("\n========== SUMMARY ==========")

print("Total comparisons:", number_of_tests)
print("Bonferroni threshold:", bonferroni_threshold)

print(
    "Significant comparisons:",
    results_df["Significant"].sum()
)

print("\nSignificant features by gas:")

print(
    results_df.groupby("Gas")["Significant"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 strongest within-gas differences:")

print(
    results_df.sort_values(
        "KS_Statistic",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\nSaved: outputs/gas_wise_drift_results.csv")
print("\n========== VALIDATION COMPLETED ==========")
