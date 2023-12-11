import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
base_dir = os.path.dirname(os.path.abspath(__file__))

# Normalizing the data
def normalize(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    return (data - mean) / std

# sampling the data
def sample_data(sample_size):
    # Set the random seed to get consistent results
    np.random.seed(0)

    # Given parameters
    theta = np.array([3, 1, 2]).reshape(-1, 1)  # Reshape to make it a column vector
    sigma_squared = 2

    # Generate x1 and x2 from normal distributions
    x1 = np.random.normal(3, np.sqrt(4), sample_size)
    x2 = np.random.normal(-1, np.sqrt(4), sample_size)

    # Generate noise for y from normal distribution
    noise = np.random.normal(0, np.sqrt(sigma_squared), sample_size)
    noise = noise.reshape(-1, 1)  # Reshape to make it a column vector

    # Calculate y using the given theta values
    X = np.column_stack((np.ones(sample_size), x1, x2))
    Y = np.dot(X, theta) + noise
    return X, Y

def shuffle_data(X, Y):
    indices = np.random.permutation(len(X))
    return X[indices], Y[indices]

sample_X, sample_Y = sample_data(1000000)

# Define hypothesis function
def h(theta, X):
    return np.dot(X, theta)

# Define cost(loss) function
def J(theta, X, Y):
    m = len(Y)
    diff = h(theta, X) - Y
    # mean of squared sum
    J = (1 / (2 * m)) * np.sum(diff ** 2)
    return J

# Define stohastic gradient descent function
def SGD(X, Y, r, k=1000, alpha = 0.001, eps=1e-9):
    # Initialize parameters
    X, Y = shuffle_data(X, Y)
    theta = np.zeros((X.shape[1], 1))
    m = len(Y)
    batches = m//r
    iterations = 0
    converged = False
    theta0_history = []
    theta1_history = []
    theta2_history = []
    last_k_loss = []
    last_k_loss_sum = 0
    next_k_loss = []
    next_k_loss_sum = 0
    while not converged:
        X, Y = shuffle_data(X, Y)

        for i in range(batches):
            iterations += 1
            X_b = X[i*r:(i+1)*r]
            Y_b = Y[i*r:(i+1)*r]
            H = h(theta, X_b)
            gradient = np.dot(X_b.T,(H-Y_b))/(2*r)
            theta -= alpha * gradient
            curr_loss = J(theta, X_b, Y_b)
            theta0_history.append(theta[0][0])
            theta1_history.append(theta[1][0])
            theta2_history.append(theta[2][0])
            if iterations <= k:
                last_k_loss.append(curr_loss)
                last_k_loss_sum += curr_loss
                continue
            elif (iterations > k) and (iterations <= 2*k):
                next_k_loss.append(curr_loss)
                next_k_loss_sum+=curr_loss
                continue
            if abs(last_k_loss_sum/k - next_k_loss_sum/k) < eps * abs(last_k_loss_sum/k) or iterations > 1000000:
                converged = True
                break
            last_k_loss_sum = last_k_loss_sum - last_k_loss.pop(0) + next_k_loss[0]
            last_k_loss.append(next_k_loss[0])
            next_k_loss_sum = next_k_loss_sum - next_k_loss.pop(0) + curr_loss
            next_k_loss.append(curr_loss)
    return theta, curr_loss, iterations, theta0_history, theta1_history, theta2_history

def plot_theta(theta0_history, theta1_history, theta2_history):
    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the values in 3D space
    ax.plot(theta0_history, theta1_history, theta2_history, color='red')

    # Set labels for each axis
    ax.set_xlabel('$\\theta_{0}$')
    ax.set_ylabel('$\\theta_{1}$')
    ax.set_zlabel('$\\theta_{2}$')
    ax.set_title('Movement of Parameters')
    plt.show()

# Preprocessing the given data
given_csv_file_path = os.path.join(base_dir, 'q2test.csv')
data = pd.read_csv(given_csv_file_path)
# Extract X (features) and Y (target) columns
X = data.iloc[:, :2].values
Y = data.iloc[:, 2].values
Y = Y.reshape(-1, 1)
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # Add term for interceptum

batches = [(1, 1000), (100, 1000), (10000, 100), (1000000, 1)]
for batch_size in batches:
    theta, loss, iterations, theta0_history, theta1_history, theta2_history = SGD(sample_X, sample_Y, batch_size[0], batch_size[1], 0.001)
    print("batch size and k: ", batch_size, "iterations: ", iterations, "loss: ", loss, "theta: ", theta)
    print("Error with test data: ", J(theta, X, Y))
    # theta_original = np.array([[3], [1], [2]])
    plot_theta(theta0_history, theta1_history, theta2_history)
    print("Error with sample data: ", J(theta, sample_X, sample_Y))