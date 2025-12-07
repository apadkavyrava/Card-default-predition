# Project Overview

This project focuses on building a machine learning model capable of predicting credit card default risk based on users’ demographic information, payment history, and usage behavior.
The goal is to identify high-risk customers early, helping financial institutions reduce losses and improve credit decision processes.

This is a supervised classification problem, where the target variable indicates whether a customer defaulted or not.

# Data Overview

The dataset is complete, with no missing or invalid values.

It contains two main types of features:
1. Demographic/Profile Data
Includes attributes such as age, gender, education, marital status, and credit limit.
2. Payment History & Usage Data
Covers payment delays over the last 6 months, bill amounts, and amounts paid.

Target Variable:
Indicates whether the customer defaulted (binary classification).

# Visualization Overview

To better understand the patterns in customer behavior and their relationship to credit default, several exploratory visualizations were created:

### 1. Default Distribution

A bar chart shows a high class imbalance, with the majority of customers falling into the non-default category. This highlights the need for techniques like resampling or class-weight adjustments during modeling.

### 2. Demographic Analysis

Multiple bar charts explore how default rates vary across:
- Gender
- Education level
- Marital status
- Age groups
These plots provide insights into whether certain demographic segments are associated with higher default risk.

### 3. Credit Limit Distribution

A histogram compares credit limits between default and non-default customers.
Non-default customers typically have higher credit limits, suggesting a possible link between creditworthiness and credit exposure.

### 4. Payment Delay History

Boxplots across six months of repayment history (PAY_0 to PAY_5) show a strong pattern:
- Defaulting customers consistently have higher delay values (late payments).
- Non-default customers cluster around on-time or early payments.
- This confirms payment behavior as a key predictive feature.

### 5. Correlation Heatmap

A heatmap visualizes correlations across all dataset features:
- Strong positive correlations appear among bill amounts and payment amounts across months.
- Payment delay variables also show interconnected patterns.
- The map helps identify redundancy and guides feature selection.
<img width="1413" height="528" alt="Screenshot 2025-12-08 at 00 27 32" src="https://github.com/user-attachments/assets/37c3d193-c843-4e13-8939-b0be9b300923" />
<img width="1232" height="728" alt="Screenshot 2025-12-08 at 00 28 09" src="https://github.com/user-attachments/assets/c9e892d9-d344-4e24-b9b3-024abb03fc24" />

# Model Selection

To determine the most suitable approach for predicting credit card default risk, I evaluated four different classification algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Each model was trained using a balanced and normalized version of the dataset to ensure fair comparison.

Model Performance Summary

<img width="574" height="127" alt="Screenshot 2025-12-08 at 00 31 01" src="https://github.com/user-attachments/assets/22553bef-8085-497c-a4a6-52cbd720ddb8" />

Among the four trained models, Random Forest and XGBoost achieved the strongest performance based on classification metrics.
Given their superior performance, these two models were selected for further fine-tuning and optimization.

# Models turning

I evaluated overfitting using cross-validation, and the accuracy remained stable across folds.

### Feature Selection
Feature selection was performed with RFE and RFECV.
RFE suggested the top 10 features.
RFECV recommended keeping all 23.
Models trained with all 23 features performed better (accuracy 0.70 vs. 0.67), so the full feature set was used.

### Hyperparameter Tuning
Each model was optimized using RandomizedSearchCV, which improved performance by efficiently exploring parameter combinations.

# Final Model Comparison

To compare the performance of the two best models — XGBoost and Random Forest — I evaluated their ROC curves and AUC scores.

Both models show very similar performance, with XGBoost slightly ahead:

<img width="719" height="626" alt="Screenshot 2025-12-08 at 00 34 56" src="https://github.com/user-attachments/assets/8d703974-ce08-48ce-b81a-f1b94e0f36e7" />

XGBoost AUC: 0.7715
Random Forest AUC: 0.7700

The ROC curves confirm that both models effectively separate default vs. non-default cases, with XGBoost holding a marginal advantage. This makes XGBoost the preferred final model for this classification task.

## Model Storage

The final trained model is saved as a **.pkl** file and stored in the **`Model/`** folder for easy loading and deployment.




