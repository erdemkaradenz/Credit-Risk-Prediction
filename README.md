# 🏦 End-to-End Credit Risk Prediction (Machine Learning)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Sklearn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## 📌 Project Overview
In the financial sector, minimizing "Charged Off" (defaulted) loans is crucial for profitability. This project simulates a real-world Data Science task: building a machine learning model to predict whether a customer will repay their loan or default.

I processed a large dataset (**Lending Club Data**), performed extensive cleaning, and built a **Random Forest Classifier** that saved the simulated bank approximately **$19.5 Million** in potential losses compared to a baseline model.

## 🛠️ Tech Stack & Workflow
* **Data Processing:** Pandas, NumPy (Handling 1.3M+ rows, Missing Value Imputation)
* **Visualization:** Matplotlib, Seaborn (EDA, Feature Importance)
* **Machine Learning:** Scikit-Learn (Logistic Regression, Random Forest)
* **Metrics:** Recall, Precision, F1-Score, Confusion Matrix

## 📊 Key Results
| Model | Recall (Detecting Defaults) | False Negatives (Missed Risk) | Business Impact |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 75% | 54,137 | Baseline |
| **Random Forest** | **76%** | **51,617** | **Saved ~$25M (Prevented Loss)** |

> **Business Insight:** The most critical predictors of default were **FICO Score** (`last_fico_range_low`) and **Debt Settlement History**. Income (`annual_inc`) was surprisingly less impactful than credit history.

## 📂 Project Structure
* `notebooks/`: 
    * `1_data_cleaning.ipynb`: ETL process (Null handling, Encoding).
    * `2_modeling.ipynb`: Model training & Evaluation.
* `models/`: Stores the serialized model and scaler files.
    * *Note: The `.pkl` files are not uploaded to GitHub due to size limits. You can generate them locally by running `notebooks/2_modeling.ipynb`.*
