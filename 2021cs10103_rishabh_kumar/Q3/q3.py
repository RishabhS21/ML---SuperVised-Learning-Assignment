import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Preprocessing the data
base_dir = os.path.dirname(os.path.abspath(__file__))
x_csv_file_path = os.path.join(base_dir, 'logisticX.csv')
y_csv_file_path = os.path.join(base_dir, 'logisticY.csv')
X = np.array(pd.read_csv(x_csv_file_path).values)
Y = np.array(pd.read_csv(y_csv_file_path).values)
# Normalizing the data
def normalize(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    return (data - mean) / std

X = normalize(X)
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # Add term for interceptum

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define hypothesis function(sigmoid here)
def h(theta, X):
    return 1 / (1 + np.exp(-np.dot(X, theta)))

# Define log-likelihood function
def log_likelihood(theta, X, Y):
    m = len(Y)
    H = sigmoid(np.dot(X, theta))
    return (1 / m) * np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H))

# Define hessian function
def hessian(theta, X):
    m = len(X)
    H = sigmoid(np.dot(X, theta))
    return (1 / m) * np.dot(X.T, np.dot(np.diag(H * (1 - H)), X))

# Define logistic Regression
def logistic_regression(X, Y):
    eps = 1e-6
    theta = np.zeros((X.shape[1], 1))
    m = len(Y)
    iterations = 0
    converged = False
    prev_cost = log_likelihood(theta, X, Y)
    while not converged:
        iterations += 1
        prev_theta = theta
        H = sigmoid(np.dot(X, theta))
        # performing newton update
        gradient = np.dot(X.T, (H - Y)) / m
        diag_mat = np.diag((H * (1 - H)).flatten())
        # print(temp)
        hessian = np.dot(X.T, np.dot(diag_mat, X)) / m
        theta -= np.dot(np.linalg.inv(hessian), gradient)
        curr_cost = log_likelihood(theta, X, Y)
        if abs(curr_cost-prev_cost) < eps or iterations > 10000:
            converged = True
        prev_cost = curr_cost
    return theta, curr_cost, iterations

theta, curr_cost, iterations = logistic_regression(X, Y)
print("Learned theta :", theta)
# Separate data points based on labels
X_label_0 = [X[i] for i in range(len(Y)) if Y[i] == 0]
X_label_1 = [X[i] for i in range(len(Y)) if Y[i] == 1]
# plot the function h with learned model theta

X_label_0 = np.array(X_label_0)
X_label_1 = np.array(X_label_1)
# Define x1 range for plotting
x1_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)

# Calculate corresponding x2 values using the decision boundary equation
x2_range = -(theta[0] + theta[1] * x1_range) / theta[2]

# Plot the line
plt.plot(x1_range, x2_range, color='green', label="Decision Boundary")
plt.scatter(X_label_0[:, 1], X_label_0[:, 2], color='red', label='Label 0')
plt.scatter(X_label_1[:, 1], X_label_1[:, 2], color='blue', label='Label 1')

plt.xlabel(r'$X_{1}$')
plt.ylabel(r'$X_{2}$')
plt.legend()
plt.title('Logistic Regression')
plt.show()