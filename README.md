# Customer Churn Prediction

## Overview
Customer churn, or customer attrition, refers to when a customer ceases their relationship with a company or service provider. In today's highly competitive business environment, retaining customers is a critical factor for long-term success. Predicting customer churn can help organizations take proactive steps to retain customers, thus minimizing revenue loss. This project aims to build a machine learning model that can predict whether a customer will churn based on their demographic, account, and service-related data.

## Problem Statement
The goal of this project is to develop a classification model that predicts whether a customer will churn. Using demographic data (such as gender, senior citizen status, and tenure), along with information about the services they use (such as internet service, phone service, and online security), this project attempts to build a model that helps the company identify customers who are at a high risk of churning. By predicting customer churn, the company can proactively design retention strategies to keep these customers, thereby improving customer satisfaction and reducing financial loss.

## Dataset Information
The dataset used for this project is `Customer_data.xlsx`. The data dictionary provides details for each variable:

* **customerID**: Unique ID for the customer
* **gender**: Gender of the customer
* **SeniorCitizen**: Whether the customer is a senior citizen (0: No, 1: Yes)
* **Partner**: Whether the customer has a partner (Yes/No)
* **Dependents**: Whether the customer has dependents (Yes/No)
* **tenure**: Number of months the customer has stayed with the company
* **PhoneService**: Whether the customer has phone service (Yes/No)
* **MultipleLines**: Whether the customer has multiple lines (Yes/No)
* **InternetService**: Customer’s internet service provider (DSL, Fiber optic, No)
* **OnlineSecurity**: Whether the customer has online security add-on (Yes/No)
* **OnlineBackup**: Whether the customer has online backup add-on (Yes/No)
* **DeviceProtection**: Whether the customer has device protection add-on (Yes/No)
* **TechSupport**: Whether the customer has tech support add-on (Yes/No)
* **StreamingTV**: Whether the customer has streaming TV add-on (Yes/No)
* **StreamingMovies**: Whether the customer has streaming movies add-on (Yes/No)
* **Contract**: Type of contract the customer has (Month-to-month, One year, Two year)
* **PaperlessBilling**: Whether the customer uses paperless billing (Yes/No)
* **PaymentMethod**: Customer’s payment method (Electronic check, Mailed check, etc.)
* **MonthlyCharges**: The amount charged to the customer monthly
* **TotalCharges**: The total amount charged to the customer
* **Churn**: Whether the customer churned (Yes/No)

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