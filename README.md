# Customer Churn Prediction using Machine Learning

## Project Overview

Customer retention is a critical challenge across many industries, particularly telecommunications where acquiring a new customer is generally more expensive than retaining an existing one.

This project uses machine learning techniques to identify customers who are likely to leave a telecommunications provider (churn). The objective was to build and evaluate predictive models capable of identifying at-risk customers and to understand which customer characteristics are most strongly associated with churn behaviour.

---

## Business Problem

Customer churn directly impacts revenue, profitability and forecasting accuracy.

The aim of this project was to answer the following question:

> Can historical customer account information be used to accurately identify customers who are likely to churn?

A successful model could be used to:

- Target specific customers for retention campaigns
- Improve retention
- Reduce marketing expenditure
- Support data-driven decision making

---

## Dataset

### IBM Telco Customer Churn Dataset

Source:

[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)

Original dataset supplied by:

> IBM Sample Data Sets

The dataset contains customer demographic information, service subscriptions, billing information and customer tenure data together with a target variable indicating whether a customer churned.

### Dataset Features

Examples include:

| Category | Example Features |
|-----------|-----------|
| Demographics | Gender, Senior Citizen, Partner, Dependents |
| Services | Internet Service, Online Security, Streaming TV |
| Billing | Monthly Charges, Total Charges, Payment Method |
| Account Information | Contract Type, Tenure |
| Target Variable | Churn |

---

## Technology Stack Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook
- GitHub

---

# Data Preparation

## Data Cleaning

Several preprocessing steps were required before modelling could begin.

### Missing Values

The `TotalCharges` column contained blank values.

These values were converted to numeric format and replaced with appropriate values following investigation.

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```
Blank values were treated as '0' and replaced as such.

### Categorical Encoding

Machine learning algorithms require numerical input.

Categorical fields were converted using one-hot encoding.

```python
pd.get_dummies(
    df,
    drop_first=True
)
```

### Train/Test Split

The dataset was split into:

- 80% Training Data
- 20% Testing Data

This ensured model evaluation was performed against unseen data.

---

# Exploratory Data Analysis

## Correlation Analysis

A correlation matrix was produced to investigate relationships between customer attributes and churn behaviour.

### Key Findings

- Tenure showed a negative relationship with churn.
- Contract length showed a negative relationship with churn.
- Monthly charges showed a positive relationship with churn.

### Correlation Matrix

![correlation-matrix](assets/correlation-matrix.png)

---

## Feature Importance Investigation

Feature importance analysis was conducted using a Random Forest model.

### Most Important Features

- Tenure
- Monthly Charges
- Contract Type
- Internet Service
- Payment Method

### Feature Importance Chart

![feature-importance](assets/feature-importance.png)

---

# Machine Learning Models

Multiple models were implemented and compared.

---

## Model 1: Logistic Regression

### Why Logistic Regression?

Logistic Regression was selected as the baseline model because:

- Simple and interpretable
- Common benchmark for binary classification
- Provides a useful performance baseline

### Results

| Metric | Value |
|----------|----------|
| Accuracy | 0.82 |
| Precision (Churn) | 0.69 |
| Recall (Churn) | 0.60 |
| F1 Score (Churn) | 0. |

### Logistic Regression Results

![Logistic-results](assets/logistic-results.png)

---

## Model 2: Random Forest

### Why Random Forest?

Random Forest was selected because:

- Handles non-linear relationships
- Robust to noise
- Naturally provides feature importance scores
- Requires minimal feature scaling

### Initial Results

| Metric | Value |
|----------|----------|
| Accuracy | 0.79 |
| Precision (Churn) | 0.66 |
| Recall (Churn) | 0.47 |
| F1 Score (Churn) | 0.55 |

Although overall accuracy remained strong, recall for churn cases was relatively low.

This suggested the model was struggling to identify customers who were actually leaving the service.

### Random Forest Results

![random-forest-results](assets/random-forest-results.png)

---

# Hyperparameter Optimisation

Grid Search Cross Validation was used to identify improved model parameters.

Parameters investigated included:

- Number of estimators
- Maximum tree depth
- Minimum samples per split
- Minimum samples per leaf
- Number of features considered
- Class weighting

Example search space:

```python
param_grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"]
}
```

---

# Feature Selection

Feature importance scores were used to remove approximately 30% of the least informative variables.

Removed examples included:

- PhoneService
- StreamingMovies
- StreamingTV
- DeviceProtection
- Some Payment Method indicators

This reduced model complexity and focused learning on higher-value features.

---

# Final Model

## Advanced Random Forest

The final model used:

```text
n_estimators = 800
max_depth = 10
min_samples_split = 5
min_samples_leaf = 1
max_features = sqrt
class_weight = balanced
```

### Why Class Weighting?

The dataset contained more non-churn customers than churn customers.

Using balanced class weights encouraged the model to pay greater attention to churn cases.

---

# Final Results

| Metric | Class 0 | Class 1 |
|----------|----------|----------|
| Precision | 0.91 | 0.55 |
| Recall | 0.77 | 0.80 |
| F1 Score | 0.83 | 0.65 |

Overall Accuracy:

```text
78%
```

---

## Key Improvement

The most significant improvement was:

| Model | Churn Recall |
|----------|----------|
| Logistic Regression | 0.51 |
| Basic Random Forest | 0.46 |
| Final Optimised Random Forest | 0.80 |

The final model successfully identified approximately 80% of customers who churned.

While overall accuracy reduced slightly, this trade-off significantly increased business value.

---

## Confusion Matrix

![final confusion matrix](assets/confusion-matrix-final.png)

---

## Normalised Confusion Matrix

![normalised matrix](assets/normalised-confusion-matrix.png)

---

## ROC Curve

![ROC](assets/roc-curve.png)

---

# Business Interpretation

The final model prioritised identifying customers likely to churn.

From an operational perspective:

- More at-risk customers were detected
- Greater opportunity for intervention
- Improved usefulness for a retention campaign

Although this increased false positives, the cost of contacting a customer unnecessarily is generally lower than losing a customer entirely.

---

# Lessons Learned

This project demonstrated:

- Data cleaning and preprocessing techniques
- Handling missing values
- Feature engineering
- Supervised machine learning
- Model evaluation
- Hyperparameter optimisation
- Feature selection
- Business-focused model assessment

Most importantly, it demonstrated that the most accurate model is not always the most useful model.

Selecting evaluation metrics aligned to the business objective is often more important than maximising overall accuracy.

---

# References

IBM Sample Data Sets

https://community.ibm.com

Kaggle Dataset

https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

Scikit-Learn Documentation

https://scikit-learn.org/stable/

Pandas Documentation

https://pandas.pydata.org/docs/
