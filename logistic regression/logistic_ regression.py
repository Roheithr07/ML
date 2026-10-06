import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Import CSV
data = pd.read_csv("data.csv")

print(data.head())

# Input and output
X = data[["YearsExperience"]]
y = data["Salary"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Actual vs Predicted
print("\nActual vs Predicted Salary")
for actual, predicted in zip(y_test, y_pred):
    print("Actual:", actual, "Predicted:", predicted)

# MSE
mse = mean_squared_error(y_test, y_pred)

# RMSE
rmse = np.sqrt(mse)

# R2
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# 10 years
salary_10 = model.predict([[10]])
print("\nPredicted Salary for 10 Years Experience:", salary_10[0])

# 15 years
salary_15 = model.predict([[15]])
print("Predicted Salary for 15 Years Experience:", salary_15[0])