# Bank Loan Approval Predictor
This application predicts the whether the candidate is applicable of getting loan from the bank based on the following details : - 

1 - Applicant Income

2 - Co Applicant Income

3 - Loan Amount

4 - Loan Amount Term

5 - Credit History

6 - No. of Dependents

7 - Gender

8 - Marital Status

9 - Education

10 - Employment Type

11 - Property Location Type

Output variable :
1 - Loan Approval Status (Approved/Rejected)

## Application Link

Use the [link](https://bank-loan-pro.herokuapp.com/) to run the web application on Browser.

## Methodology
### Dataset
The Dataset is selected from kaggle.

![plot](./wine_app/ml_models/CountPlot.png)

### Data and Feature Engineering
#### Handling Missing Values
The missing values were replaced with mode of the column. Only the 'Loan Amount' column was imputed with median value so as to take into account the outliers too. 

#### Handling Class Imbalance
Class imbalance was handled using the SMOTE technique.

### New Features Imputation
Further few relevant features were added to the feature list :- 
#Total income
TotalIncome = ApplicantIncome  + CoapplicantIncome
 
#Monthly income
monthly_amount = LoanAmount/Loan_Amount_Term

#The income left after the monthly amount has been paid
left_income = TotalIncome - monthly_amount*1000



### Model Training

Multiple Machine Learning Models were trained to predict the class of the wine being tested using the above mentioned features. 

The following models were used (shown along with confusion matrix evaluation) :-

1 - LightGB Model

Accuracy = 81.49%

![plot](./lgb1.JPG)

2 - XGBoost Model

Accuracy = 83.07%

![plot](./xgb1.JPG)

3 - LightGB Model with KFold (K=5)

Accuracy = 86.26%

![plot](./lgb2.JPG)

4 - XGBoost Model with KFold (K=5)

Accuracy = 83.89%

![plot](./xgb2.JPG)

5 - Ensembling 

Expression used -> y_pred = alpha*(y_xgb_kfold) + (1 - alpha)*(y_lgb_kfold)

where y_xgb_kfold is y_pred from model 3 and y_lgb_kfold is y_pred from model 4

alpha selected -> 0.5

Accuracy = 83.89%

![plot](./en.JPG)

Therefore ensemble model provides best confusion metrics with accuracy also being 83.89%. Hence selected.


### Application Deployment
The model is deployed using Django framework with templating for basic frontend. The Django application is deployed on Heroku.


### Screenshots

Input Form Page

![plot](./ss1.JPG)

![plot](./ss2.JPG)

Output Page

![plot](./ss3.JPG)

