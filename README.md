# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries for numerical operations, data handling, and preprocessing.

2.Load the startup dataset (50_Startups.csv) using pandas.

3.Extract feature matrix X and target vector y from the dataset.

4.Convert feature and target values to float and reshape if necessary.

5.Standardize X and y using StandardScaler.

6.Add a column of ones to X to account for the bias (intercept) term.

7.Initialize model parameters (theta) to zeros.

8.Perform gradient descent to update theta by computing predictions and adjusting for error.

9.Input a new data point, scale it, and add the intercept term.

10.Predict the output using learned theta, then inverse-transform it to get the final result.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Hashwatha M
RegisterNumber: 212223240051
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
    X = np.c_[np.ones(len(X1)), X1]
    # Initialize theta with zeros
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    # Perform gradient descent
    for _ in range(num_iters):
        # calculate prediction
        predictions = (X).dot(theta).reshape(-1,1)
        # calculate errors
        errors = (predictions - y).reshape(-1,1)
        # Update theta using gradient descent
        theta -= learning_rate * (1/len(X1)) * X.T.dot(errors)
    return theta
data = pd.read_csv('50_Startups.csv')
print(data.head())
```
```
# Assuming the last column is your target variable 'y' and the preceding columns a
X = (data.iloc[1:, :-2].values)
print(X)
X1 = X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
```
```
#Learn model parameters
theta = linear_regression(X1_Scaled, Y1_Scaled)
#Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
print("Name:Hashwatha M")
print("Reg No:212223240051")
```
## Output:
## Data Information
![image](https://github.com/user-attachments/assets/e8a7e9e0-6dea-4d1d-8aa4-23022b5aeb33)
## Values of X and Y
![image](https://github.com/user-attachments/assets/cf073ebe-0942-40c6-b8f1-4756a2d3895b)
![image](https://github.com/user-attachments/assets/c1de35b6-a5f5-4a53-901f-a8b4eb045fdb)
## Predicted Value
![image](https://github.com/user-attachments/assets/b2465e36-f767-428c-94ee-ea3bdd16f150)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
