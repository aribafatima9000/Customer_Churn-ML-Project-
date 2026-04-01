# Customer Churn Prediction using Random Forest

## 📌 Project Overview
Customer churn is a critical challenge for subscription-based businesses. This project aims to predict whether a customer will discontinue their service based on their demographics, usage patterns, and support interactions. By identifying at-risk customers early, businesses can take proactive measures to improve retention.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn
* **Model:** Random Forest Classifier

## 📊 Dataset Description
The dataset contains **10,000 customer records** with **32 features**, including:
* **Demographics:** Gender, Age, Country, City.
* **Usage:** Tenure months, Monthly charges.
* **Support:** Complaint types and feedback.
* **Target:** `Churn` (Yes/No).

## 🚀 Machine Learning Pipeline

### 1. Data Cleaning & Preprocessing
* **Missing Value Imputation:** Handled ~2,000 missing values in the `complaint_type` column by categorizing them as "Unknown".
* **Feature Dropping:** Removed non-predictive columns like `customer_id` to reduce noise.
* **Categorical Encoding:** Applied Label Encoding to transform categorical variables into a machine-readable format.

### 2. Exploratory Data Analysis (EDA)
* Analyzed churn distribution across genders (Found approx. 10% churn rate for both Male and Female).
* Visualized feature correlations to understand key drivers of customer attrition.

### 3. Model Development
* Split data into **80% training** and **20% testing** sets.
* Trained a **Random Forest Classifier** to handle complex non-linear relationships.

### 4. Evaluation
* **Accuracy:** 90%
* **Metrics:** Generated Classification Report (Precision, Recall, F1-Score) and Confusion Matrix.
* **Feature Importance:** Identified `Feature_13` and `Feature_10` as the top predictors of churn.

## 📈 Key Insights
* The model achieves a high accuracy of **90%**, making it reliable for business decision-making.
* Certain specific features (Feature 13 & 10) have a significantly higher impact on churn, suggesting that business interventions should focus on these areas.

## 📂 How to Run
1. Clone this repository.
2. Install dependencies: `pip install pandas scikit-learn matplotlib`.
3. Run the Jupyter Notebook: `jupyter notebook Untitled115.ipynb`.

---
