# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Load car data, split it into training and testing sets, and create polynomial features for Polynomial Regression.
2. Model Training: Train both Linear and Polynomial Regression models using the training data.
3. Prediction and Evaluation: Predict car prices for the test data using both models, and calculate the Mean Squared Error (MSE) for each.
4. Visualization: Plot the actual vs predicted prices for both models to visually compare their performance.

## Program:
```c
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: HAMZA FAROOQUE
RegisterNumber: 212223040054
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'encoded_car_data.csv'
df = pd.read_csv(file_path)

# Select relevant features and target variable
X = df[['enginesize', 'horsepower', 'citympg', 'highwaympg']]  # Features
y = df['price']  # Target variable

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Evaluate Linear Regression
print("Linear Regression:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_linear))
print("R-squared:", r2_score(y_test, y_pred_linear))

# 2. Polynomial Regression
poly = PolynomialFeatures(degree=2)  # Change degree for higher-order polynomials
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate Polynomial Regression
print("\nPolynomial Regression:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_poly))
print("R-squared:", r2_score(y_test, y_pred_poly))

# Visualize Results
plt.figure(figsize=(10, 5))

# Plot Linear Regression Predictions
plt.scatter(y_test, y_pred_linear, label='Linear Regression', color='blue', alpha=0.6)

# Plot Polynomial Regression Predictions
plt.scatter(y_test, y_pred_poly, label='Polynomial Regression', color='green', alpha=0.6)

plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2)  # Ideal Line
plt.title("Linear vs Polynomial Regression Predictions")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.legend()
plt.show()

```

## Output:

![image](https://github.com/user-attachments/assets/c1b7b8be-dd2b-4e8a-8461-3ef63f12e0eb)


![image](https://github.com/user-attachments/assets/7c300772-fd2a-4883-8639-debe79946412)



## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
