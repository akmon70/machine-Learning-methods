"""
General steps of linear regression with Normal Equation
1- Data Preprocessing: Start by collecting and preprocessing the data that you want to use for training the model. This involves tasks like cleaning the data, handling missing values, and feature scaling.

2- Formulate the Model: In linear regression, you want to find the values of the coefficients that best fit the data. To do this, you need to formulate a linear equation that relates the input variables (also called features or independent variables) to the output variable (also called the dependent variable). The equation has the form: y = X * theta + epsilon, where y is the output variable, X is the matrix of input variables, theta is the vector of coefficients or weights, and epsilon is the error term.

3- Solve for the Coefficients: To find the values of the coefficients that minimize the sum of squared errors between the predicted values and the actual values, we can solve the normal equation: theta = (X^T * X)^-1 * X^T * y. Here, X^T is the transpose of X, and (X^T * X)^-1 is the inverse of the matrix product of X^T and X.

4- Evaluate Model: Finally, we evaluate the performance of the model on a test set. We can calculate metrics like the root mean squared error (RMSE) or the coefficient of determination (R^2) to assess how well the model generalizes to new data.
"""
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt


class LinearRegression:
    """A linear regression model using the normal equation to fit the data."""

    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the data.

        Parameters:
            X (ndarray): input matrix of shape (n_samples, n_features)
            y (ndarray): target vector of shape (n_samples,)
        """
        # add a column of ones to X for the intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # calculate the normal equation coefficients
        self.theta = pinv(X.T @ X) @ X.T @ y

    def predict(self, X):
        """
        Predict target values for new data.

        Parameters:
            X (ndarray): input matrix of shape (n_samples, n_features)

        Returns:
            ndarray: predicted target values of shape (n_samples,)
        """
        # add a column of ones to X for the intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # predict y using the fitted coefficients
        y_pred = X @ self.theta

        return y_pred


# -------------------- test -------------
# 5 Training data with 3 parameters (variables)
# X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 7, 9]])
# y = np.array([6, 15, 24, 7, 100])

X = 100 * np.random.rand(100, 1)
y = 40 + 30 * X + 400 * np.random.randn(100, 1)

model = LinearRegression()
model.fit(X, y)

X_new = np.array([[1.4]])
y_pred = model.predict(X_new)
print(y_pred)
Y = model.predict(X)
plt.scatter(X, y)
plt.scatter(X_new, y_pred, color='red')
plt.plot(X, Y, color='green')
plt.show()
