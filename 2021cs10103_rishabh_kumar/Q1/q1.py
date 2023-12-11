import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 0 for rows and 1 for coloumns
# Preprocessing the data
base_dir = os.path.dirname(os.path.abspath(__file__))
x_csv_file_path = os.path.join(base_dir, 'linearX.csv')
y_csv_file_path = os.path.join(base_dir, 'linearY.csv')
X = np.array(pd.read_csv(x_csv_file_path).values)
Y = np.array(pd.read_csv(y_csv_file_path).values)
# Normalizing the data
def normalize(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    return (data - mean) / std

X = normalize(X)
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # Add term for interceptum
# Define hypothesis function
def h(theta, X):
    return X.dot(theta)

# Define cost(loss) function
def J(theta, X, Y):
    m = len(Y)
    diff = h(theta, X) - Y
    # mean of squared sum
    J = (1 / (2 * m)) * np.sum(diff ** 2)
    return J

# Gradient Descent
def GD(X, Y, alpha = 0.1):
    # Initialize parameters
    theta = np.zeros((X.shape[1], 1))
    m = len(Y)
    #alpha is the Learning rate
    iterations = 0
    prev_cost = J(theta, X, Y)
    eps = 1e-9
    converged = False
    loss_history = []
    theta0_history = []
    theta1_history = []
    while not converged:
        iterations += 1
        H = h(theta, X)
        gradient = np.dot(X.T,(H-Y))/(2*m)
        # print(X.T.shape, H.shape, gradient.shape)
        theta -= alpha * gradient
        current_cost = J(theta, X, Y)
        if abs(current_cost - prev_cost) < eps or iterations > 100000:
            converged = True
        prev_cost = current_cost
        loss_history.append(current_cost)
        theta0_history.append(theta[0][0])
        theta1_history.append(theta[1][0])
    
    # Hypothesis plot
    plt.title('Input Dataset')
    plt.scatter(X[:,1:],Y,label='Given Data', color='blue')
    plt.plot(X[:,1:],H,label='Hypothesis', color='red')
    plt.xlabel(' Acidity(X) ')
    plt.ylabel(' Density(Y) ')
    # hypothesis_path = os.path.join(base_dir, 'results/hypothesis.png')
    # plt.savefig(hypothesis_path)
    plt.legend()
    plt.show()
    return iterations, theta, current_cost, loss_history, theta0_history, theta1_history

# Gradient Descent plot
def plot_gd(loss_history, theta0_history, theta1_history):
    fig, ax1 = plt.subplots()
    # plot thetas over time
    ax1.plot(theta0_history, label='$\\theta_{0}$', linestyle='--', color='blue')
    ax1.plot(theta1_history, label='$\\theta_{1}$', linestyle='-', color='blue')
    ax1.set_xlabel('Iterations'); ax1.set_ylabel('$\\theta$')

    # plot loss function over time
    ax2 = ax1.twinx()
    ax2.plot(loss_history, label='Loss function', color='red')
    ax2.set_title('Values of $\\theta$ and $J(\\theta)$ over iterations')
    ax2.set_ylabel('$J(\\theta)$')
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center')

    fig.tight_layout()
    plt.show()

# Draw a 3-dimensional mesh showing the error function (J(θ)) on z-axis and the parameters in the x − y plane.
def plot_mesh(loss_history, theta0_history, theta1_history):
    # A meshgrid for theta0 and theta1
    theta0_vals = np.linspace(0, 2, 100)
    theta1_vals = np.linspace(-1, 1, 100)
    Theta0, Theta1 = np.meshgrid(theta0_vals, theta1_vals)

    # filling the cost values for each combination of theta0 and theta1
    Cost = np.zeros_like(Theta0)
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            theta = np.array([[theta0_vals[i]], [theta1_vals[j]]])
            Cost[i, j] = J(theta, X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surface = ax.plot_surface(Theta0, Theta1, Cost, cmap='viridis')

    # labels and title
    ax.set_xlabel('$\\theta_{0}$')
    ax.set_ylabel('$\\theta_{1}$')
    ax.set_zlabel('Cost J($\\theta$)')
    ax.set_title('3-dimensional mesh for Error Function')

    # Initialising scatter plot for displaying current parameter values during GD
    scatter = ax.scatter([], [], [], color='red', s=50)

    # Function to update scatter plot data and pause for visualization
    def update_scatter(i):
        scatter._offsets3d = (theta0_history[:i+1], theta1_history[:i+1], loss_history[:i+1])
        plt.pause(0.2)  # Pause for 0.2 seconds for visualization

    # Animate the scatter plot over iterations
    for i in range(iterations):
        update_scatter(i)
    plt.show()
    
# the contours of the error function 
def plot_contours(loss_history, theta0_history, theta1_history):
    # A meshgrid for theta0 and theta1
    theta0_vals = np.linspace(0, 2, 20)
    theta1_vals = np.linspace(-1, 1, 20)
    Theta0, Theta1 = np.meshgrid(theta0_vals, theta1_vals)

    # cost values for each combination of theta0 and theta1
    Cost = np.zeros_like(Theta0)
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            theta = np.array([[theta0_vals[i]], [theta1_vals[j]]])
            Cost[i, j] = J(theta, X, Y)

    fig, ax = plt.subplots()

    contour = ax.contour(Theta0, Theta1, Cost, cmap='viridis')

    # Set labels and title
    ax.set_xlabel('$\\theta_{0}$')
    ax.set_ylabel('$\\theta_{1}$')
    ax.set_title('Contour Plot of Error Function $J(\\theta)$')

    # Initialize scatter plot for displaying current parameter values during GD
    scatter = ax.scatter([], [], color='red', s=1)

    # Create a function to update scatter plot data and pause for visualization
    def update_scatter(i):
        scatter.set_offsets(np.column_stack((theta0_history[:i+1], theta1_history[:i+1])))
        plt.pause(0.2)  # Pause for 0.2 seconds for visualization

    # Animate the scatter plot over iterations
    for i in range(iterations):
        update_scatter(i)

    plt.show()

if __name__ == "__main__":
    iterations, theta, cost, loss_history, theta0_history, theta1_history = GD(X, Y)
    plot_gd(loss_history, theta0_history, theta1_history)
    plot_mesh(loss_history, theta0_history, theta1_history)
    plot_contours(loss_history, theta0_history, theta1_history)
    print("Number of Iterations to converge: ", iterations)
    print("Final Cost: ", cost)
    print("Final Theta: ", theta)