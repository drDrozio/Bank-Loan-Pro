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



#### Scaling
The data was scaled using the Standard Scalar from Scikit learn library.

### Decomposition
PCA was applied but no such improvement was found in the model performance with PCA. Therefore not used.

### Model Training

Multiple Machine Learning Models were trained to predict the class of the wine being tested using the above mentioned features. 

The following models were used (shown along with confusion matrix evaluation) :-

1 - LGB

Accuracy = 81.49%

![plot](./lgb1.jpeg)

2 - XGB

Accuracy = 83.07%

![plot](./xgb1.jpeg)

3 - 

| Model with Hyperparameters                                               | Score             | Score w Balancing |
| ------------------------------------------------------------------------ | ----------------- |------------------ |
| Logistic Regression (Unoptimized)                                        | (60.32%, 56.44%)  | (44.07%, 41.48%)  |
| Logistic Regression (C=1, multiclass='ovr',solver='newton-cg')           | (60.60%, 56.63%)  | (69.56%, 68.37%)  |
| Logistic Regression (C=36, multiclass='multinomial',solver='newton-cg')  | (62.46%, 57.20%)  | (92.43%, 92.04%)  |
| Support Vector Classifier (kernel='poly',C=100000)                       | (30.53%, 33.34%)  | (97.39%, 96.40%)  |
| Support Vector Classifier (kernel='linear',C=1)                          | (60.13%, 55.87%)  | (79.74%, 42.23%)  |

Therefore it was seen that balancing operation improved model performance. Hence further hyperparameter tuning was done with graphical study to obtain optimum parameters for Support Vector Classifier. 

Best fit model -> SVC(kernel='linear',C=100,decision_function_shape='ovo',gamma=0.01)
with score -> (99.72%, 99.81%)


### Application Deployment
The model is deployed using Django framework with templating for basic frontend. The Django application is deployed on Heroku.

Flowchart of application process :-

![plot](./Flowchart.png)

### Screenshots

![plot](./Opening1.PNG)

![plot](./Opening2.PNG)

![plot](./Opening3.PNG)

![plot](./Result.PNG)
