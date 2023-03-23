
# Importing the required libraries
import numpy as np
from numpy.linalg import pinv, inv
import matplotlib.pyplot as plt

# Generating random training dataset
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Adding bias term to X
X_b = np.c_[np.ones((100, 1)), X]

# Calculating the best-fit regression line
theta_best = pinv(X_b.T @ X_b) @ (X_b.T) @ (y)

# Predicting the y values for the training data
y_predict = X_b @ theta_best

# Plotting the training data and the linear regression line
plt.scatter(X, y)
plt.plot(X, y_predict, color='red')
plt.show()
