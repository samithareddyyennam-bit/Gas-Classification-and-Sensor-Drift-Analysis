# Gas Classification and Sensor Drift Analysis Using Machine Learning

## 1. Project Overview

This project develops a machine learning system to classify six different gases using readings collected from a gas sensor array.

It also investigates sensor drift by comparing sensor feature distributions across different measurement batches.

The system uses Logistic Regression for gas classification and Streamlit to provide an interactive web application for predictions and analysis.

## 2. Problem Statement

Gas sensor readings can change over time because of sensor drift and variations in measurement conditions. These changes can reduce the accuracy of machine learning models.

The objective of this project is to classify different gases from sensor readings and analyze how changes in sensor measurements affect model performance.

## 3. Objectives

- Classify six different gas categories.
- Process and analyze gas sensor readings.
- Compare multiple machine learning algorithms.
- Investigate sensor drift across measurement batches.
- Develop an interactive web application for gas prediction.
- Evaluate model performance using classification metrics.

## 4. Dataset Description

**Dataset:** Gas Sensor Array Drift Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 224

Dataset characteristics:

- Total samples: 13,910
- Sensor features: 128
- Gas categories: 6
- Measurement batches: 10

### Gas Categories

1. Acetaldehyde
2. Acetone
3. Ammonia
4. Ethanol
5. Ethylene
6. Toluene

## 5. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- VS Code

## 6. Project Methodology

### Step 1: Data Collection

Downloaded the UCI Gas Sensor Array Drift Dataset and extracted its ten batch files.

### Step 2: Data Preprocessing

- Loaded the sensor data.
- Extracted 128 sensor features.
- Encoded gas labels.
- Applied MinMaxScaler normalization.
- Divided the data according to measurement batches.

### Step 3: Exploratory Data Analysis

Analyzed gas distribution, batch distribution, feature distributions, missing values, correlations and feature skewness.

### Step 4: Model Training

Evaluated three machine learning algorithms:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

### Step 5: Model Selection

Used batches 1–7 for training and batches 8–9 for validation.

Logistic Regression with C=10 achieved the best validation results.

### Step 6: Final Model Evaluation

Retrained the selected model using batches 1–9 and evaluated it on the independent batch 10 test set.

### Step 7: Sensor Drift Analysis

Used the Kolmogorov–Smirnov (KS) statistical test to investigate differences in sensor feature distributions.

Performed both overall and gas-wise drift comparisons.

### Step 8: Web Application Development

Developed a Streamlit application to upload sensor readings, predict gas categories, display model evaluation results and visualize sensor drift.

## 7. Model Validation Results

| Model | Validation Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression (C=10) | 82.46% | 0.7852 |
| Random Forest | 81.41% | 0.7470 |
| Logistic Regression (C=1) | 77.36% | 0.7568 |
| Logistic Regression (C=0.1) | 71.99% | 0.6771 |
| SVM | 71.34% | 0.6558 |

Selected model: Logistic Regression (C=10).

## 8. Final Model Performance

Final evaluation was performed using batch 10.

| Metric | Result |
|---|---:|
| Accuracy | 65.25% |
| Weighted Precision | 66.40% |
| Weighted Recall | 65.25% |
| Macro F1-score | 0.6441 |
| Weighted F1-score | 0.6441 |

### Class-wise F1-score

| Gas | F1-score |
|---|---:|
| Acetaldehyde | 0.44 |
| Acetone | 0.82 |
| Ammonia | 0.43 |
| Ethanol | 0.61 |
| Ethylene | 0.87 |
| Toluene | 0.70 |

## 9. Sensor Drift Analysis

The KS statistical test was used to compare feature distributions between batches 1–9 and batch 10.

Overall drift analysis:
- 128 out of 128 features showed statistically significant distribution differences at p < 0.05.
- Mean KS statistic: 0.1836.

Gas-wise drift analysis:
- 768 comparisons were performed across six gases and 128 features.
- 757 comparisons (98.57%) were statistically significant after Bonferroni correction.

These results indicate substantial distribution changes. They do not establish sensor ageing as the only cause; variations in concentration and measurement conditions may also contribute.

## 10. Streamlit Application Features

The web application provides:

1. Sensor CSV file upload.
2. Gas classification predictions.
3. Prediction confidence estimates.
4. Prediction result download.
5. Final model confusion matrix.
6. Model comparison visualization.
7. Overall sensor drift analysis.
8. Interactive gas-wise sensor drift analysis.

## 11. Project Structure

```text
Gas_Classification_ML/
│
├── dataset/
│   └── raw/
│       ├── batch1.dat
│       ├── batch2.dat
│       ├── ...
│       └── batch10.dat
│
├── models/
│   ├── final_gas_classifier.pkl
│   ├── final_scaler.pkl
│   └── label_encoder.pkl
│
├── outputs/
│   ├── gas_sensor_dataset.csv
│   ├── validation_comparison.csv
│   ├── sensor_drift_results.csv
│   ├── gas_wise_drift_results.csv
│   ├── final_confusion_matrix.png
│   └── evaluation_plots/
│
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── eda.py
│   ├── visualization.py
│   ├── drift_analysis.py
│   ├── drift_by_gas.py
│   ├── model_tuning.py
│   └── final_training.py
│
├── app.py
├── requirements.txt
└── README.md
```

## 12. How to Run the Project

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the project folder

```bash
cd Gas_Classification_ML
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the environment (Windows PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the application

```bash
python -m streamlit run app.py
```

## 13. Limitations

- Model performance decreases on the later measurement batch.
- Some gas categories are more difficult to classify than others.
- Sensor distribution changes can affect prediction reliability.
- The model has not been validated on a separate real-world sensor deployment.
- Prediction confidence values are model estimates, not guaranteed probabilities of correctness.

## 14. Future Scope

- Improve classification performance under sensor drift.
- Investigate domain adaptation and drift compensation techniques.
- Evaluate additional machine learning and deep learning algorithms.
- Develop real-time sensor data integration.
- Improve model reliability using additional measurement conditions.
- Deploy the application as an accessible web service.

## 15. Conclusion

This project demonstrates the application of machine learning to gas classification using sensor array data.

Logistic Regression with C=10 was selected based on validation performance and achieved 65.25% accuracy on the independent batch 10 test set.

The sensor drift analysis identified substantial distribution changes across measurement batches, including comparisons within individual gas categories.

The developed Streamlit application provides an interactive platform for gas prediction, model evaluation and sensor drift visualization.

## 🎥 Project Demo

Watch the live demonstration of our Gas Classification and Sensor Drift Analysis project:

▶️ [Click here to watch the project demo on YouTube] https://youtu.be/swR2b8inVGA