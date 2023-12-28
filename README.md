# Customer-Churn-Prediction-for-orange-Telecom-Company
Developed a machine learning model to predict customer churn for a Telecom company, with the goal of reducing customer attrition and improving customer retention strategies. 

![](orange.png)

[🚀 Go to web app](https://orangechurn.streamlit.app/)

real Dataset from Orange.

## 📖 Table of Contents

- [📍  Data collection](#data-collection)
- [📦 Data preprocessing](#data-preprocessing)
- [📊 Dashboard-and-report-Power-Bi](#dashboard-and-report-power-bi)
- [📂 Model building](#model-building)
- [⚙️ Model evaluation](#model-evaluation)
- [🚀 Getting Started](#-getting-started)

## 📍 Overview

This is a README file for the project of developing a machine learning model to predict customer churn for a Telecom company. The goal of this project is to reduce customer attrition and improve customer retention strategies by identifying the factors that influence customer churn and the customers who are most likely to churn.

The project consists of the following steps:
---
##  Data collection:
The data used for this project is obtained from the Orange.inc, which contains information about 5000 customers of a Telecom company, such as demographic, account, service, and billing details. The data also includes a binary variable indicating whether the customer has churned or not.
- Data exploration: The data is explored using descriptive statistics and visualizations to understand the distribution, correlation, and relationship of the variables. Some of the key findings are:
    - The **churn rate** of the customers is **14.4%**.
    - Customers who have **month-to-month contracts**, **no online security**, **no tech support**, and **fiber optic internet service** are more likely to churn.
    - Customers who have **longer tenure**, **multiple lines**, **online backup**, **device protection**, and **paperless billing** are less likely to churn.
---
## Data preprocessing:
The data is preprocessed by performing the following tasks:
    - Handling missing values: There are 11 missing values in the TotalCharges column, which are imputed by using the median value of the column.
    - Encoding categorical variables: The categorical variables are encoded using one-hot encoding or label encoding, depending on whether they are nominal or ordinal.
    - Scaling numerical variables: The numerical variables are scaled using standardization to have zero mean and unit variance.
    - Splitting the data: The data is split into training (80%) and testing (20%) sets, with stratification to preserve the class balance.
## Dashboard-and-report-Power-Bi:
Below are a potential KPIs created for a telecom dashboard:
- Churn Rate:  The percentage of customers who have churned.
- Average Call Duration: The average duration of calls across all customers.
- Customer Service Calls per Customer: The average number of customer service calls per customer.
- International Call Percentage: The percentage of total calls that are international.
- Revenue from International Calls: The total revenue generated from international calls.
- Average Revenue per User (ARPU):The average revenue generated per user.
- Voice Mail Subscription Rate:The percentage of customers who have subscribed to the voice mail plan.

In a telecom churn analysis, we can create different dashboards to gain insights into various aspects of customer churn. In our case, we will create an Overview Dashboard and a Financial Impact Dashboard based on our data.
- 1-Overview Dashboard:
    - Provides a high-level summary of key metrics and KPIs related to churn.
    - Includes total customer base, churn rate, revenue, and customer service metrics.
    - Visualizations such as pie charts, cards, and gauges may be used for a quick overview.
- 2-Financial Impact Dashboard:
    - Examines the financial impact of churn on revenue and profitability.
    - Compares the revenue loss due to churn against other revenue streams.
    - Visualizations may include waterfall charts, line charts, and tables.
After analyzing the data and discovering patterns, I will ask some important questions related to the data:
- Question1: What is the overall churn rate, and how does it vary across different states?
- Question2: How does the presence of international plans impact customer churn?
- Question3: What is the relationship between the number of customer service calls and churn?
- Question4: How do total charges correlate with customer churn?
- Question: Are there specific patterns in call duration (day, evening, night) for customers who churn?

## Model building:
Several machine learning models are built using different algorithms, such as logistic regression, decision tree, random forest, support vector machine, and gradient boosting. The models are trained on the training set and evaluated on the testing set using various metrics, such as accuracy, precision, recall, f1-score, and roc_auc_score. The best model is selected based on the highest roc_auc_score, which is a measure of how well the model can distinguish between the positive (churned) and negative (not churned) classes.
## Model evaluation: 
The best model is evaluated on the testing set using a confusion matrix, a classification report, and an ROC curve. The results are as follows:
    - Confusion matrix: The confusion matrix shows the number of true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN) predicted by the model.

                      | Predicted Positive | Predicted Negative |
    |-----------------|--------------------|--------------------|
    | Actual Positive | 229 (TP)           | 286 (FN)           |
    | Actual Negative | 97 (FP)            | 997 (TN)           |

- Classification report: The classification report shows the precision, recall, f1-score, and support for each class.

    |       | Precision | Recall | F1-score | Support |
    |-------|-----------|--------|----------|---------|
    | Positive | 0.70      | 0.44   | 0.54     | 515   |
    | Negative | 0.78      | 0.91   | 0.84     | 1094  |
    | Macro avg | 0.74     | 0.68    | 0.69    | 1609  |
    
- ROC curve: The ROC curve shows the trade-off between the true positive rate (TPR) and the false positive rate (FPR) for different thresholds of the model. The area under the curve (AUC) is a measure of how well the model can distinguish between the classes.

## ROC curve

The ROC curve shows that the model has an AUC of 0.82, which means that it has a good ability to separate the positive and negative classes.

- Model interpretation: The model is interpreted by using feature importance and partial dependence plots to understand how each feature affects the prediction. Some of the key insights are:
- The most important features for predicting customer churn are **tenure**, **contract**, **monthly charges**, **online security**, and **tech support**.
- Customers with **shorter tenure**, **higher monthly charges**, **month-to-month contracts**, **no online security**, and **no tech support** have a higher probability of churning.
- Customers with **longer tenure**, **lower monthly charges**, **two-year contracts**, **online security**, and **tech support** have a lower probability of churning.

- Model deployment: The model is deployed as a web application using Streamlit. The web application allows users to enter their customer information and get a prediction of whether they are likely to 
  churn or not. The web application also provides some recommendations on how to retain customers who are at risk of churning.

The web application can be accessed at  https://orangechurn.streamlit.app/


