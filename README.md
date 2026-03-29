# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the dataset by removing unnecessary columns, converting categorical data, and standardizing features and target.
2.Split the dataset into training and testing sets.

3.Train the SGD Regressor model using the training data and predict prices for the test data.

4.Evaluate the model using MSE and R² score, then visualize actual vs predicted prices using a scatter plot .

## Program:
```
#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import pandas as pd
data = pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())

#Data preprocessing
#Dropping unnecessary columns and handling categorical variables
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

#Splitting the data info features and target variable
X = data.drop('price', axis=1)
y = data['price']

#Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1, 1))

#Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating the SGD Regressor model
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)

#Fitting the model on the training data
sgd_model.fit(X_train, y_train.ravel())

#Making predictions
y_pred = sgd_model.predict(X_test)

#Evaluating model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("="*50)
print('Name: SARANYA R')
print('Reg No: 212225040384')
print(f"MSE: {mse:.4f}")
print(f"R2: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.4f}")
print("="*50)

#Print model coefficients
print("Model coefficients:")
print("coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

#Visualizing actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGDRegressor")
plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)],
         color='red')
plt.show()
```

## Output:
<img width="548" height="104" alt="image" src="https://github.com/user-attachments/assets/db3ed2e8-b166-4dc6-ae5e-9362c031ed17" />
<img width="1002" height="580" alt="image" src="https://github.com/user-attachments/assets/154ba4b8-f8d4-4b8b-90dd-6e871728254f" />



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
