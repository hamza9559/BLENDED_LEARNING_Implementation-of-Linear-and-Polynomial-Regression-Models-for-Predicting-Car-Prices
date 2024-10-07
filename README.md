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
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: HAMZA FAROOQUE
RegisterNumber:  212223040054
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

data = {
    'Age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Mileage': [5000, 15000, 25000, 35000, 45000, 55000, 65000, 75000, 85000, 95000],
    'Price': [20000, 18500, 17500, 16500, 15500, 14500, 13500, 12500, 11500, 10500]
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Feature variables (Age and Mileage)
X = df[['Age', 'Mileage']]

# Target variable (Price)
y = df['Price']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Linear Regression -----
# Create and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict car prices using the linear model
linear_pred = linear_model.predict(X_test)

# Calculate the MSE for linear regression
linear_mse = mean_squared_error(y_test, linear_pred)

# ----- Polynomial Regression -----
# Create polynomial features (degree=2 for quadratic)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Create and train the Polynomial Regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predict car prices using the polynomial model
poly_pred = poly_model.predict(X_test_poly)

# Calculate the MSE for polynomial regression
poly_mse = mean_squared_error(y_test, poly_pred)

# ----- Output Results -----
print(f"Linear Regression Predictions: {linear_pred}")
print(f"Linear Regression MSE: {linear_mse}")
print(f"Polynomial Regression Predictions: {poly_pred}")
print(f"Polynomial Regression MSE: {poly_mse}")

# ----- Visualization -----
# Plot the actual vs predicted prices for both models
plt.scatter(y_test, linear_pred, color='blue', label='Linear Predictions')
plt.scatter(y_test, poly_pred, color='green', label='Polynomial Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Actual Values Line')

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices (Linear and Polynomial)')
plt.legend()
plt.show()

```

## Output:
```
Linear Regression Predictions: [11426.72413793 18698.27586207]
Linear Regression MSE: 22341.334720570405
Polynomial Regression Predictions: [11493.32249859 18764.87422272]
Polynomial Regression MSE: 35101.47144456938
```

![image](https://github.com/user-attachments/assets/9d53be33-8b48-482a-b91a-e4fa099db58b)


## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
