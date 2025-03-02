import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("C:/Users/preet/Downloads/FMLA1Q1Data_train.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Add a column of ones to X for the bias term
X = np.c_[np.ones((X.shape[0], 1)), X]

# Load dataset
data = pd.read_csv("C:/Users/preet/Downloads/FMLA1Q1Data_test.csv")
x_test = data.iloc[:, :-1].values
y_test = data.iloc[:, -1].values

# Add a column of ones to X for the bias term
x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]

def compute_mse_with_w(w_ml, X_val, y_val):
    y_val_pred = X_val @ w_ml
    
    # Compute MSE on validation set
    mse = mean_squared_error(y_val, y_val_pred)
    return mse

#Question 1
# Compute weights using the analytical solution
w_ml = np.linalg.inv(X.T @ X) @ X.T @ y

print("The least squares solution w_ML is:", w_ml)
print("mse on test set = ", compute_mse_with_w(w_ml,x_test,y_test))

#Question 2
# Computing weights using gradient descent
learning_rate = 0.001  
num_iterations = 1000
tolerance = 1e-10

# Initializing weights to zeros
w_t = np.zeros(X.shape[1])

norm_values = []

# Gradient descent loop
for t in range(num_iterations):
    w_t_prev=w_t
    gradient = X.T @ (X @ w_t - y)
    w_t = w_t - learning_rate * gradient # Update weights
    
    # norm of the difference between w_t and w_ml
    norm = np.linalg.norm(w_t - w_ml)
    norm_values.append(norm)
    
    # Check for convergence
    if np.linalg.norm(w_t-w_t_prev) < tolerance:
        print(f'Converged after {t+1} iterations')
        break

print("The gradient descent solution is", w_t)

print("mse on test set = ", compute_mse_with_w(w_t,x_test,y_test))

# Plot: ∥wt − wML∥2 as a function of iterations
plt.plot(range(len(norm_values)), norm_values)
plt.xlabel('Iteration (t)')
plt.ylabel('norm ∥wt - wML∥2')
plt.title('Convergence of Gradient Descent')
plt.grid(True)
plt.show()

#Question 3
# computing weights using stochastic gradient descent
learning_rate = 0.001  
batch_size = 100       
num_iterations = 10000  
tolerance = 1e-10

w_t = np.zeros(X.shape[1])
w_t_values = []

norm_values = []
norm_avg_values = []

# Stochastic Gradient Descent Loop
for t in range(num_iterations):
    batch_indices = np.random.choice(X.shape[0], batch_size, replace=False) #random sample of 100 data points
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]
    
    gradient = X_batch.T @ (X_batch @ w_t - y_batch)
    w_t_prev=w_t
    w_t_values.append(w_t.copy())

    w_t = w_t - learning_rate * gradient
    w_t_avg = np.mean(w_t_values, axis=0) # average of all w_t values so far
    
    # norm of the difference between w_t_avg and w_ml
    norm_avg = np.linalg.norm(w_t_avg - w_ml)
    # norm of the difference between w_t and w_ml
    norm = np.linalg.norm(w_t - w_ml)
    
    norm_values.append(norm)
    norm_avg_values.append(norm_avg)
    
    # Check for convergence
    if np.linalg.norm(w_t-w_t_prev) < tolerance:
        print(f'Converged after {t+1} iterations')
        break

# ∥wt − wML∥2 as a function of iterations
plt.plot(range(len(norm_values)), norm_values)
plt.xlabel('Iteration (t)')
plt.ylabel('∥wt - wML∥2')
plt.title('Variance of weights in Stochastic Gradient Descent')
plt.grid(True)
plt.show()

# Plot ∥wt_avg − wML∥2 as a function of iterations
plt.plot(range(len(norm_avg_values)), norm_avg_values)
plt.xlabel('Iteration (t)')
plt.ylabel('∥wt_avg - wML∥2')
plt.title('Convergence of avg weights in Stochastic Gradient Descent')
plt.grid(True)
plt.show()

print("The stochastic gradient descent solution is", w_t_avg)
print("mse on test set = ", compute_mse_with_w(w_t_avg,x_test,y_test))

#Question 4
# computing weights using ridge regression
learning_rate = 0.001  
num_iterations = 1000  

# Split the data into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# List of λ 
lambda_values = np.logspace(-2.5, 2.5, 10)  # From 0.001 to 1000
validation_errors = []

# Ridge Regression
def ridge_regression(X, y, x_test, y_test, learning_rate, num_iterations, lambda_reg):
    w_t = np.zeros(X.shape[1])
    # Gradient descent loop
    for _ in range(num_iterations):
        gradient = X.T @ (X @ w_t - y) + lambda_reg * w_t
        w_t = w_t - learning_rate * gradient

    y_val_pred = x_test @ w_t   #ypred on test set
    # MSE on test set
    error = mean_squared_error(y_test, y_val_pred)
    return error

# Cross-validation for various values of λ
for lambda_reg in lambda_values:
    validation_error = ridge_regression(X, y, X_val, y_val, learning_rate, num_iterations, lambda_reg)
    validation_errors.append(validation_error)

# Plot validation error as a function of λ
plt.plot(lambda_values, validation_errors, marker='o')
plt.xscale('log')  
plt.xlabel('λ (Regularization parameter)')
plt.ylabel('Mean Square Error (MSE)')
plt.title('MSE vs λ in Ridge Regression')
plt.grid(True)
plt.show()

# minimum MSE and its corresponding λ value
min_validation_error = min(validation_errors)
min_index = validation_errors.index(min_validation_error)
best_lambda = lambda_values[min_index]

print("Minimum validation error:", min_validation_error)
print("Corresponding λ value: ", best_lambda)

#computing the WR for best lambda and corresponding error
lambda_reg=best_lambda
w_t = np.zeros(X.shape[1])

# Gradient descent loop
for _ in range(num_iterations):
    gradient = X.T @ (X @ w_t - y) + lambda_reg * w_t
    w_t = w_t - learning_rate * gradient

y_val_pred = x_test @ w_t
validation_error = mean_squared_error(y_test, y_val_pred)

W_R=w_t
print("W_R for the best lambda is ",W_R)
print("The minimum MSE value for Ridge Regression is ", validation_error, ". The MSE with linear regression is 66.25642957811056. ")

#Question 5

# visualising the data
# Load dataset
data = pd.read_csv("C:/Users/preet/Downloads/FMLA1Q1Data_train.csv")

# Extract the features (X) and target values (y)
X = data.iloc[:, :-1].values  # First two columns as features
y = data.iloc[:, -1].values   # Last column as target

# Create a figure with two subplots, one for each feature vs. y
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot Feature 1 vs Target (y)
axs[0].scatter(X[:, 0], y, c='g', alpha=0.6)
axs[0].set_xlabel('Feature 1')
axs[0].set_ylabel('Target (y)')
axs[0].set_title('Feature 1 vs. Target (y)')

# Plot Feature 2 vs Target (y)
axs[1].scatter(X[:, 1], y, c='r', alpha=0.6)
axs[1].set_xlabel('Feature 2')
axs[1].set_ylabel('Target (y)')
axs[1].set_title('Feature 2 vs. Target (y)')

# Adjust layout for better visualization
plt.tight_layout()
plt.show()

def quadratic_kernel(x1, x2):
    return (np.dot(x1, x2.T) + 1) ** 2

def compute_kernel_matrix(X1, X2):
    return quadratic_kernel(X1, X2)

# Kernel Ridge Regression
def kernel_ridge_regression(X_train, y_train, X_test, lam=1e-3):
    K_train = compute_kernel_matrix(X_train, X_train)
    I = np.eye(K_train.shape[0])
    alpha = np.linalg.inv(K_train + lam * I) @ y_train
    K_test = compute_kernel_matrix(X_test, X_train) #kernel matrix between test and training data
    
    # Prediction on test data
    y_pred = K_test @ alpha
    return y_pred

lam = 0.01   # Regularization parameter

# Kernel Ridge Regression with Quadratic Kernel and predict on test data
y_test_pred_quadratic = kernel_ridge_regression(X_train, y_train, x_test, lam=lam)
test_error_quadratic = mean_squared_error(y_test, y_test_pred_quadratic)
print("Quadratic Kernel Ridge Regression Error: ", test_error_quadratic)

y_test_pred_ls = x_test @ w_ml
test_error_ls = mean_squared_error(y_test, y_test_pred_ls)
print("Least Squares Regression Error: ",test_error_ls)