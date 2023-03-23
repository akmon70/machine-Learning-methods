# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

# Generating random training dataset
X = 2 * np.random.rand(100, 2)
y = 4 + 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100)

# Adding bias term to X
X_b = np.c_[np.ones((100, 1)), X]

# Setting the hyperparameters
eta = 0.1
n_iterations = 1000
m = 100

# Initializing the theta values
theta = np.random.randn(3, 1)

# Performing gradient descent
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T @ (X_b @ theta - y.reshape(-1, 1))
    theta = theta - eta * gradients

# Predicting the y values for the training data
y_predict = X_b @ (theta)

# Plotting the training data and the linear regression plane
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
X1, X2 = np.meshgrid(X[:, 0], X[:, 1])
Y = theta[0] + theta[1]*X1 + theta[2]*X2
ax.plot_surface(X1, X2, Y, alpha=0.5)
plt.show()
