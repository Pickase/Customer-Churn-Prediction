# Customer Churn Prediction

## Overview
Customer churn, or customer attrition, refers to when a customer ceases their relationship with a company or service provider. In today's highly competitive business environment, retaining customers is a critical factor for long-term success. Predicting customer churn can help organizations take proactive steps to retain customers, thus minimizing revenue loss. This project aims to build a machine learning model that can predict whether a customer will churn based on their demographic, account, and service-related data.

## Problem Statement
The goal of this project is to develop a classification model that predicts whether a customer will churn. Using demographic data (such as gender, senior citizen status, and tenure), along with information about the services they use (such as internet service, phone service, and online security), this project attempts to build a model that helps the company identify customers who are at a high risk of churning. By predicting customer churn, the company can proactively design retention strategies to keep these customers, thereby improving customer satisfaction and reducing financial loss.


## Methodology and Deliverables

The project involved the following key phases:

1.  **Data Exploration and Preprocessing**: Analyze the dataset for trends, missing values, and outliers. Perform necessary cleaning, feature engineering, and encoding to prepare the data for modeling.
2.  **Exploratory Data Analysis (EDA)**: Visualize the data to identify patterns, relationships between variables, and insights relevant to customer churn.
3.  **Model Building**: Develop and train various classification models to predict customer churn.
4.  **Model Evaluation**: Evaluate the performance of the models using appropriate metrics such as accuracy, confusion matrix, classification report, F1-score, and ROC-AUC score.
5.  **Hyperparameter Tuning**: Tune the hyperparameters of the models to improve performance.
6.  **Prediction on New Data**: Utilize the final model to predict churn on new customer data.

The final model aims to balance accuracy with interpretability.

## Tools Required
The project uses the following tools and libraries:

* **Python**
* **Libraries**:
    * pandas
    * numpy
    * matplotlib.pyplot
    * seaborn
    * joblib
    * sklearn (scikit-learn) for model selection (train\_test\_split, GridSearchCV), various classifiers (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, SVC), and metrics (accuracy\_score, confusion\_matrix, classification\_report, f1\_score, roc\_auc\_score).
    * xgboost (XGBClassifier)
    * sklearn.preprocessing (StandardScaler, LabelEncoder, OneHotEncoder)
    * sklearn.compose (ColumnTransformer)
    * sklearn.pipeline (Pipeline)
* **Development Environment**: Jupyter Notebook or any IDE suitable for running Python code.

## How to Run the Project
1.  Ensure you have Python and all the required libraries installed.
2.  Open the `ML-Project-2-PranavJoshi-Draft.ipynb` Jupyter Notebook.
3.  Place the `Customer_data.xlsx` file in the appropriate directory as referenced in the notebook, or update the path to the dataset within the notebook.
4.  Run all cells in the notebook sequentially to execute the data exploration, cleaning, model building, evaluation, and prediction steps.

## Video Link

A video explaining the analysis for this project can be found in the "Video Link" section within the `ML-Project-2-PranavJoshi-Draft.ipynb` notebook.
