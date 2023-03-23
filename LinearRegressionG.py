"""
General steps of linear regression with Gradient Descent
1- Data Preprocessing: Start by collecting and preprocessing the data that you want to use for training the model. This involves tasks like cleaning the data, handling missing values, and feature scaling.

2- Initialize Model Parameters: In linear regression, you want to find the values of the coefficients that best fit the data. To do this, you need to initialize the values of the coefficients (also called weights) randomly or to some predefined values.

3- Define the Cost Function: The cost function measures how well the model fits the data. In linear regression, the most commonly used cost function is the mean squared error (MSE). This function calculates the average of the squared differences between the predicted values and the actual values.

4- Gradient Descent: Gradient descent is an optimization algorithm that updates the weights of the model to minimize the cost function. The algorithm works by iteratively updating the weights in the direction of the negative gradient of the cost function.

5- Update Weights: In each iteration, we update the weights by subtracting the product of the learning rate and the gradient of the cost function with respect to the weights. The learning rate is a hyperparameter that controls the step size at each iteration.

6- Iterate Until Convergence: We repeat the steps 4 and 5 until the cost function reaches a minimum or until a maximum number of iterations is reached. At each iteration, the cost function should decrease until it reaches a minimum value.

7- Evaluate Model: Finally, we evaluate the performance of the model on a test set. We can calculate metrics like the root mean squared error (RMSE) or the coefficient of determination (R^2) to assess how well the model generalizes to new data.
"""
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tol=0.0001):
        """
        Initialize the linear regression model with configurable learning rate, maximum number of iterations,
        and convergence tolerance.
        hyperparameters:
        :param learning_rate: The learning rate to use for gradient descent.
        :param max_iterations: The maximum number of iterations to use for gradient descent.
        :param tol: The convergence tolerance to use for gradient descent.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol

    def fit(self, X, y):
        """
        Fit the linear regression model to the input data using gradient descent.

        :param X: The input data to use for fitting the model, shape (n_samples, n_features).
        :param y: The target values to use for fitting the model, shape (n_samples,).
        """
        n_samples, n_features = X.shape

        # Initialize the model parameters to zeros and add a bias term to the input data
        self.theta = np.zeros((n_features + 1, 1))
        X = np.hstack((np.ones((n_samples, 1)), X))
        y = y.reshape(-1, 1)

        # Perform gradient descent to optimize the model parameters
        for i in range(self.max_iterations):
            # Make predictions using the current model parameters
            h = X @ self.theta

            # Compute the error between the predicted and actual values
            errors = h - y

            # Compute the gradient of the cost function with respect to the model parameters
            gradient = (X.T @ errors) / n_samples

            # Update the model parameters using the gradient and learning rate
            self.theta -= self.learning_rate * gradient

            # Check for convergence
            if np.max(np.abs(gradient)) < self.tol:
                break

    def predict(self, X):
        """
        Use the fitted linear regression model to make predictions on new data.

        :param X: The input data to use for making predictions, shape (n_samples, n_features).
        :return: The predicted target values, shape (n_samples,).
        """
        n_samples, n_features = X.shape

        # Add a bias term to the input data and make predictions using the current model parameters
        X = np.hstack((np.ones((n_samples, 1)), X))
        return (X @ self.theta)


# -------------------- test -------------
# 5 Training data with 3 parameters (variables)
# X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 7, 9]])
# y = np.array([6, 15, 24, 7, 100])

X = 100 * np.random.rand(100, 1)
y = 40 + 30 * X + 400 * np.random.randn(100, 1)


model = LinearRegression(learning_rate=0.01, max_iterations=10000, tol=0.0001)
model.fit(X, y)

# X_new = np.array([[2, 6, 4]])
X_new = np.array([[1.4]])
y_pred = model.predict(X_new)
print(y_pred)
Y = model.predict(X)
plt.scatter(X, y)
plt.scatter(X_new, y_pred, color='red')
plt.plot(X, Y, color='green')
plt.show()
