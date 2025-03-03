import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Data Import
dataset = pd.read_csv("C:/Users/preet/Downloads/Assignment2.data", sep='\t')
print(dataset.head(10))
num_entries = len(dataset["StockPrice"])
price_data = pd.DataFrame({"time_index": range(num_entries), "price": dataset.StockPrice})

# Task 1
target_values = np.array(price_data.price).reshape(-1, 1)
time_values = np.array(price_data.time_index).reshape(-1, 1)
product1 = np.matmul(np.transpose(time_values), time_values)
product2 = np.matmul(np.transpose(time_values), target_values)
slope_coeff = np.matmul(np.linalg.inv(product1), product2)  # solution using matrix algebra

print("Computed Slope:", slope_coeff)
slope_value1 = slope_coeff[0]
residual_errors = target_values - time_values * slope_coeff
sum_squared_err = sum([error**2 for error in residual_errors])
print("Total Squared Error:", sum_squared_err)

# Angle Search
angles = [angle for angle in range(0, 65, 5)]  # list of angles to try
sse_list = []
best_angle = 0
min_error = 10 * sum_squared_err

for angle in angles:
    slope_guess = math.tan(angle * math.pi / 180)
    errors = target_values - time_values * slope_guess
    sse = sum([err**2 for err in errors])
    sse_list.append(sse)
    if sse <= min_error:
        best_angle = angle
        min_error = sse

plt.plot(angles, sse_list)
plt.xlabel("Angle (degrees)")
plt.ylabel("Total Squared Error")
plt.title("Total Squared Error vs Angle")
plt.show()

optimal_slope1 = math.tan(best_angle * math.pi / 180)
print("Optimal angle =", best_angle, "degrees; Slope:", optimal_slope1)

# Linear Regression using Scikit-Learn
model = LinearRegression().fit(time_values, target_values)
print("Scikit-Learn Slope:", model.coef_)
slope_value2 = model.coef_[0]

# Comparing Different Slopes
plt.plot(time_values, target_values, 'bo', label='Actual Prices')  
plt.plot(time_values, time_values * slope_value1, 'g-', label='Matrix Method')  
plt.plot(time_values, time_values * optimal_slope1, 'r-', label='Angle Search') 
plt.plot(time_values, time_values * slope_value2, 'c-', label='Scikit-Learn')  

plt.ylabel('Price')
plt.xlabel('Time (days)')
plt.legend(loc='upper left')
plt.title('Comparison of Slope Estimates')
plt.show()

# Task 2

# Proposed model: y = b1*x + b2*sin(wx)
# Here, b1 and b2 are found using linear regression
# The frequency w is determined by trying out various values and selecting the optimal one

time_series = np.array([i * 1.0 for i in range(num_entries)])
sin_wave1 = np.sin(time_series / 5)
sin_wave2 = np.sin(time_series / 10)
sin_wave3 = np.sin(time_series / 15)
sin_wave4 = np.sin(time_series / 20)
sin_wave5 = np.sin(time_series / 25)

features = pd.DataFrame({
    "time": time_series, 
    "sin1": sin_wave1, 
    "sin2": sin_wave2, 
    "sin3": sin_wave3, 
    "sin4": sin_wave4, 
    "sin5": sin_wave5
})

reg_model = LinearRegression().fit(features, target_values)
print("Model Coefficients:", reg_model.coef_)

# The sine wave with the largest coefficient is selected
# The resulting equation is: y = b1*x + b2*sin(0.1*x)

# Refitting using only the selected frequency
selected_features = pd.DataFrame({"time": time_series, "sin2": sin_wave2})
reg_model = LinearRegression().fit(selected_features, target_values)
residual_errors = target_values - reg_model.predict(selected_features)
sum_squared_err = sum([err ** 2 for err in residual_errors])[0]
print("Final Sum of Squared Errors (SSE):", sum_squared_err)

plt.plot(time_series, target_values, 'mo', label='Actual Data')  # Magenta for actual data
plt.plot(time_series, reg_model.predict(selected_features), 'y-', label='Fitted Model')  # Yellow for model fit
plt.ylabel('Price')
plt.xlabel('Time (days)')
plt.legend(loc='upper left')
plt.title('Model Fit vs Actual Data')
plt.show()

# Significantly lower SSE than previous methods

# Interpolation

# Dividing the dataset into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(selected_features, target_values, test_size=0.2, random_state=42)
reg_model = LinearRegression().fit(X_train, y_train)
interp_residuals = y_test - reg_model.predict(X_test)
interp_sse = sum([err ** 2 for err in interp_residuals])[0]
print("Interpolation SSE:", interp_sse)

plt.plot(X_train["time"], y_train, 'b+', label='Training Data')  # Blue for training data
plt.plot(X_test["time"], reg_model.predict(X_test), 'r+', label='Test Predictions')  # Red for test predictions
plt.ylabel('Price')
plt.xlabel('Time (days)')
plt.legend(loc='lower right')
plt.title('Interpolation: Training vs Test')
plt.show()

# Extrapolation

# Splitting the data so the first 80% is training and the remaining 20% is testing
X_train = selected_features.iloc[0:int(0.8 * num_entries)]
X_test = selected_features.iloc[int(0.8 * num_entries) + 1:]
y_train = target_values[:int(0.8 * num_entries)]
y_test = target_values[int(0.8 * num_entries) + 1:]

reg_model = LinearRegression().fit(X_train, y_train)
extrap_residuals = y_test - reg_model.predict(X_test)
extrap_sse = sum([err ** 2 for err in extrap_residuals])[0]
print("Extrapolation SSE:", extrap_sse)

plt.plot(X_train["time"], y_train, 'g+', label='Training Data')  # Green for training data
plt.plot(X_test["time"], reg_model.predict(X_test), 'm+', label='Extrapolated Predictions')  # Magenta for extrapolated predictions
plt.ylabel('Price')
plt.xlabel('Time (days)')
plt.legend(loc='lower right')
plt.title('Extrapolation: Training vs Test')
plt.show()

# Task 3

spring_data = pd.DataFrame({"time_index": range(num_entries), "position": dataset.SpringPos})

true_values = np.array(spring_data.position).reshape(-1,1)
time_array = np.array(spring_data.time_index).reshape(-1,1) * 1.0

# Modeling data using a damped oscillation equation: y = (b1 + b2*x) * sin(wx)
# The parameters b1 and b2 are determined via linear regression.
# The frequency w is found by testing different values.

time_values = np.array([i * 1.0 for i in range(num_entries)])

sin_freq1 = np.sin(time_values / 5)
sin_freq2 = np.sin(time_values / 10)
sin_freq3 = np.sin(time_values / 15)
sin_freq4 = np.sin(time_values / 20)
sin_freq5 = np.sin(time_values / 25)

# Creating a DataFrame with features and interactions
feature_data = pd.DataFrame({
    "sin_freq1": sin_freq1, 
    "sin_freq2": sin_freq2, 
    "sin_freq3": sin_freq3, 
    "sin_freq4": sin_freq4, 
    "sin_freq5": sin_freq5, 
    "time_sin_freq1": time_values * sin_freq1, 
    "time_sin_freq2": time_values * sin_freq2,
    "time_sin_freq3": time_values * sin_freq3,
    "time_sin_freq4": time_values * sin_freq4,
    "time_sin_freq5": time_values * sin_freq5
})

# Fitting a linear regression model to the full feature set
full_fit = LinearRegression().fit(feature_data, true_values)
print("Full model coefficients:", full_fit.coef_)

# Selecting the feature with the largest coefficient (sin_freq2)
selected_features = pd.DataFrame({
    "sin_freq2": sin_freq2,
    "time_sin_freq2": time_values * sin_freq2
})

# Fitting a linear regression model with the chosen feature
reduced_fit = LinearRegression().fit(selected_features, true_values)
print("Reduced model coefficients:", reduced_fit.coef_)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(time_values, true_values, 'bo', label='Observed Data')  # Blue for observed data
plt.plot(time_values, reduced_fit.predict(selected_features), 'g-', label='Fitted Model')  # Green for the fitted model

plt.ylabel('Displacement')
plt.xlabel('Time (s)')
plt.legend(loc='upper left')
plt.title('Regression Fit to Oscillating Data')
plt.grid(True)
plt.show()
