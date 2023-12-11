import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Preprocessing the data
base_dir = os.path.dirname(os.path.abspath(__file__))
x_csv_file_path = os.path.join(base_dir, 'q4x.dat')
y_csv_file_path = os.path.join(base_dir, 'q4y.dat')
X = pd.read_csv(x_csv_file_path, sep='\s+', header=None)
Y = pd.read_csv(y_csv_file_path, sep='\s+', header=None)
# Convert Pandas DataFrames to NumPy arrays
X = X.values
Y = Y.values
# Fuction to normalize the data
def normalize(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    return (data - mean) / std

X = normalize(X)
X_class_0 = [X[i] for i in range(len(Y)) if Y[i] == 'Alaska']
X_class_1 = [X[i] for i in range(len(Y)) if Y[i] == 'Canada']
X_class_0 = np.array(X_class_0)
X_class_1 = np.array(X_class_1)
# Calculate means, µ0 and µ1
mu_0 = np.mean(X_class_0, axis=0)
mu_1 = np.mean(X_class_1, axis=0)

# Calculate covariance matrix, Σ if Σ0 and Σ1 are same
m = len(Y)
X_modified = []
for i in range(m):
    if Y[i] == 'Alaska':
        X_modified.append(X[i] - mu_0)
    else:
        X_modified.append(X[i] - mu_1)
X_modified = np.array(X_modified)
sigma = np.dot(X_modified.T, X_modified)/m

# Calculate covariance matrices, Σ0 and Σ1
X_diff_0 = X_class_0 - mu_0.reshape(1, -1)
sigma_0 = np.dot(X_diff_0.T, X_diff_0)/len(X_class_0)
X_diff_1 = X_class_1 - mu_1.reshape(1, -1)
sigma_1 = np.dot(X_diff_1.T, X_diff_1)/len(X_class_1)

print("mu_0 : ", mu_0)
print("mu_1 : ", mu_1)
print("sigma : ", sigma)
print("sigma_0 : ", sigma_0)
print("sigma_1 : ", sigma_1)

# Plotting linear decision boundary
# Calculate φ
indicator_1 = 1 * (Y == 'Canada').reshape(-1, 1)
phi = np.sum(indicator_1) / m
sigma_inv = np.linalg.pinv(sigma)
sigma_0_inv = np.linalg.pinv(sigma_0)
sigma_1_inv = np.linalg.pinv(sigma_1)

def plot_linear():
    x1_vals = np.linspace(-2, 2, 2)
    # Calculate x2 values using the h function
    coefficient_term = -1 * np.dot(np.transpose(mu_1 - mu_0), sigma_inv)
    constant_term_1 = np.dot(np.dot(np.transpose(mu_1), sigma_inv), mu_1)
    constant_term_2 = np.dot(np.dot(np.transpose(mu_0), sigma_inv), mu_0)
    constant_term_3 = np.log((1 - phi) / phi)
    constant_term = (constant_term_1 - constant_term_2) / 2 + constant_term_3
    x2_vals = []
    for x1 in x1_vals:
        # Calculate x2 from h value (inverse of h function)
        x2 = -1*(constant_term+coefficient_term[0]*x1) / coefficient_term[1]
        x2_vals.append(x2)
    x2_vals = np.array(x2_vals)
    return x1_vals, x2_vals

def plot_quadratic():
    d_1 = np.sqrt(np.linalg.det(sigma_1))
    d_0 = np.sqrt(np.linalg.det(sigma_0))
    coefficient_term_1 = 0.5*(sigma_1_inv - sigma_0_inv)
    coefficient_term_2 = np.dot(mu_1.T, sigma_1_inv) - np.dot(mu_0.T, sigma_0_inv)
    constant_term = 0.5*(np.dot(np.dot(mu_1.T, sigma_1_inv), mu_1) - np.dot(np.dot(mu_0.T, sigma_0_inv), mu_0)) + np.log((1-phi*d_1)/phi*d_0)
    x1_vals = np.linspace(-3, 3, 200)
    x2_vals = np.linspace(-3, 3, 200)
    x1, x2 = np.meshgrid(x1_vals, x2_vals)
    term_1 = (x1**2)*coefficient_term_1[0][0] + x1*x2*(coefficient_term_1[0][1] + coefficient_term_1[1][0]) + (x2**2)*coefficient_term_1[1][1]
    term_2 = coefficient_term_2[0]*x1 + coefficient_term_2[1]*x2
    h = term_1 - term_2 + constant_term
    return plt.contour(x1, x2, h, levels=[0], colors='orange')


x1_vals, x2_vals = plot_linear()
# Plot the data points
plt.scatter(X_class_0[:, 0], X_class_0[:, 1], color='red', label='Alaska')
plt.scatter(X_class_1[:, 0], X_class_1[:, 1], color='blue', label='Canada')

# Plot the decision boundary
plt.plot(x1_vals, x2_vals, color='green')
plot_quadratic()
# Set labels and legend
plt.xlabel(r'$X_{1}$')
plt.ylabel(r'$X_{2}$')
plt.legend()
plt.title('Gaussian Discriminant Analysis')
# Show the plot
plt.show()