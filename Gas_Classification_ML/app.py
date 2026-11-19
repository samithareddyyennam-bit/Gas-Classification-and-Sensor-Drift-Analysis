
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

# Page configuration
st.set_page_config(
    page_title="Gas Classification System",
    page_icon="🧪",
    layout="wide"
)

# Load trained model files
@st.cache_resource
def load_model_files():
    model = joblib.load(
        MODEL_DIR / "final_gas_classifier.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "final_scaler.pkl"
    )

    encoder = joblib.load(
        MODEL_DIR / "label_encoder.pkl"
    )

    return model, scaler, encoder


model, scaler, encoder = load_model_files()

# Feature names
features = [f"feature_{i}" for i in range(1, 129)]

# Application heading
st.title("🧪 Gas Classification System")

st.write(
    "An ML-based system that identifies six types of gases "
    "using gas sensor readings and analyzes sensor drift."
)

st.divider()

# Model information
col1, col2, col3 = st.columns(3)

col1.metric("Gas Categories", "6")
col2.metric("Sensor Features", "128")
col3.metric("Final Test Accuracy", "65.25%")

st.divider()

# Upload sensor data
st.subheader("Upload Sensor Readings")

st.write(
    "Upload a CSV file containing 128 sensor features "
    "(feature_1 to feature_128)."
)

uploaded_file = st.file_uploader(
    "Choose your sensor CSV file",
    type=["csv" , "dat"]
)

if uploaded_file is not None:

    try:
        input_data = pd.read_csv(uploaded_file)

        # Check required feature columns
        missing_features = [
            feature for feature in features
            if feature not in input_data.columns
        ]

        if missing_features:
            st.error(
                f"Missing {len(missing_features)} required "
                "sensor features. Your CSV must contain "
                "feature_1 through feature_128."
            )

        else:
            # Keep only sensor features in correct order
            X = input_data[features].copy()

            # Validate numeric values
            X = X.apply(pd.to_numeric, errors="coerce")

            if X.isnull().any().any():
                st.error(
                    "Your sensor data contains missing or "
                    "non-numeric feature values."
                )

            else:
                # Scale and predict
                X_scaled = scaler.transform(X)

                predictions = model.predict(X_scaled)

                predicted_gases = encoder.inverse_transform(
                    predictions
                )

                # Add predictions to uploaded data
                results = input_data.copy()
                results["Predicted_Gas"] = predicted_gases

                # Show results
                st.success("Gas prediction completed!")

                st.subheader("Prediction Results")

                st.dataframe(
                    results[["Predicted_Gas"] + features[:5]],
                    width="stretch"
                )

                # Confidence scores
                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(X_scaled)

                    confidence = probabilities.max(axis=1) * 100

                    results["Prediction_Confidence"] = confidence

                    st.subheader("Prediction Confidence")

                    st.write(
                        f"Average confidence: "
                        f"{confidence.mean():.2f}%"
                    )

                # Download results
                csv_data = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv_data,
                    file_name="gas_prediction_results.csv",
                    mime="text/csv"
                )

    except Exception as error:
        st.error(f"Something went wrong: {error}")

else:
    st.info(
        "Upload a CSV file containing 128 sensor features "
        "to start gas classification."
    )

st.divider()

st.subheader("About This Project")

st.write(
    """
    This project uses Logistic Regression to classify six gases:

    - Acetaldehyde
    - Acetone
    - Ammonia
    - Ethanol
    - Ethylene
    - Toluene

    The model was trained on batches 1–9 and evaluated on
    the independent batch 10 test set.

    Sensor drift was investigated by comparing sensor feature
    distributions across measurement batches.
    """
)

# ==============================
# PROJECT ANALYTICS DASHBOARD
# ==============================

st.divider()
st.title("📊 Project Analytics Dashboard")

# 1. Predicted gas distribution
if uploaded_file is not None and "results" in locals():

    st.subheader("1. Predicted Gas Distribution")

    gas_counts = results["Predicted_Gas"].value_counts()

    st.bar_chart(gas_counts)

# 2. Final confusion matrix
st.subheader("2. Final Model Confusion Matrix")

confusion_path = PROJECT_DIR / "outputs" / "final_confusion_matrix.png"

if confusion_path.exists():
    st.image(
        str(confusion_path),
        caption="Final model evaluation on Batch 10",
        width="stretch"
    )
else:
    st.warning("Confusion matrix image not found.")

# 3. Model comparison
st.subheader("3. Model Comparison")

comparison_path = PROJECT_DIR / "outputs" / "validation_comparison.csv"

if comparison_path.exists():

    comparison = pd.read_csv(comparison_path)

    st.dataframe(comparison, use_container_width=True)

    st.bar_chart(
        comparison.set_index("Model")[["Validation Accuracy", "Macro F1"]]
    )
else:
    st.warning("Model comparison file not found.")


# 4. Sensor drift analysis
st.subheader("4. Sensor Drift Analysis")

drift_path = PROJECT_DIR / "outputs" / "sensor_drift_results.csv"

if drift_path.exists():

    drift_data = pd.read_csv(drift_path)

    # Sort by highest KS statistic
    drift_data = drift_data.sort_values(
        by="KS_Statistic",
        ascending=False
    )

    st.write(
        "Top 10 sensor features showing distribution changes."
    )

    # Display graph
    st.bar_chart(
        drift_data.head(10).set_index("Feature")["KS_Statistic"]
    )

    # Display table
    st.dataframe(
        drift_data.head(10),
        width="stretch"
    )

    # Explain results
    st.info(
        "A higher KS statistic indicates a larger difference "
        "between the sensor feature distributions of batches 1–9 "
        "and batch 10."
    )

else:
    st.warning("Sensor drift results file not found.")
    
# =====================================
# 5. GAS-WISE SENSOR DRIFT ANALYSIS
# =====================================

st.divider()
st.subheader("5. Gas-wise Sensor Drift Analysis")

gas_drift_path = (
    PROJECT_DIR / "outputs" / "gas_wise_drift_results.csv"
)

if gas_drift_path.exists():

    gas_drift = pd.read_csv(gas_drift_path)

    # Select gas
    gas_options = sorted(gas_drift["Gas"].unique())

    selected_gas = st.selectbox(
        "Select a gas to analyze",
        gas_options
    )

    # Filter selected gas
    selected_data = gas_drift[
        gas_drift["Gas"] == selected_gas
    ].copy()

    # Sort by KS statistic
    selected_data = selected_data.sort_values(
        by="KS_Statistic",
        ascending=False
    )

    # Calculate significant features
    significant_count = selected_data["Significant"].sum()
    total_features = len(selected_data)

    col1, col2, col3 = st.columns(3)

    col1.metric("Selected Gas", selected_gas)
    col2.metric("Significant Features", int(significant_count))
    col3.metric("Total Features", total_features)

    # Top 10 features
    st.write("Top 10 sensor features with the largest distribution changes")

    top_features = selected_data.head(10)

    st.bar_chart(
        top_features.set_index("Feature")["KS_Statistic"]
    )

    # Detailed table
    st.write("Detailed Statistical Results")

    st.dataframe(
        selected_data,
        width="stretch"
    )

    # Explanation
    st.info(
        "The KS statistic measures the difference between "
        "sensor feature distributions for the selected gas "
        "in batches 1–9 and batch 10. Higher values indicate "
        "larger distribution differences. Significant indicates "
        "whether the comparison passed the statistical threshold "
        "used in the drift analysis."
    )

else:
    st.warning("Gas-wise drift results file not found.")

