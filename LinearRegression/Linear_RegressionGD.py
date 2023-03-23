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


class LinearRegressionGD:
    """
    Linear regression model using gradient descent with normalization.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The learning rate to use in the gradient descent algorithm.
    max_iterations : int, default=1000
        The number of iterations to use in the gradient descent algorithm.
    normalize : bool, default=True
        Whether to normalize the input data.

    Attributes
    ----------
    coefficients(theta) : array, shape (n_features,)
        The learned coefficients for the linear regression model.
    mean : array, shape (n_features,)
        The mean of the input data, if `normalize=True`.
    std : array, shape (n_features,)
        The standard deviation of the input data, if `normalize=True`.

    Methods
    -------
    fit(X, y)
        Fit the linear regression model to the input data using gradient descent.
    predict(X)
        Make predictions for new input data using the learned coefficients(theta).
    """

    def __init__(self, learning_rate=0.01, max_iterations=1000, tol=0.0001, normalize=True):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol
        self.normalize = normalize
        self.theta = None
        self.mean = None
        self.std = None

    def normalize_data(self, X):
        """
        Normalize the input data using mean normalization.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input data to normalize.

        Returns
        -------
        X_norm : array, shape (n_samples, n_features)
            The normalized input data.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_norm = (X - self.mean) / self.std
        return X_norm

    def fit(self, X, y):
        """
        Fit the linear regression model to the input data using gradient descent.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input data to fit the model to.
        y : array, shape (n_samples,)
            The target values to fit the model to.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.normalize:
            X = self.normalize_data(X)
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
        Make predictions for new input data using the learned coefficients(theta).

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input data to make predictions for.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            The predicted target values for the input data.
        """
        if self.normalize:
            X = (X - self.mean) / self.std
        # Add a bias term to the input data and make predictions using the current model parameters
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        # predict
        y_pred = X @ self.theta
        return y_pred


# -------------------------- test ---------------
# create some sample data
# X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 7, 9]])
# y = np.array([6, 15, 24, 7, 100])
X = 100 * np.random.rand(100, 1)
y = 40 + 30 * X + 400 * np.random.randn(100, 1)

# create a LinearRegressionGD instance
model = LinearRegressionGD(
    learning_rate=0.1, max_iterations=1000, normalize=True)

# fit the model to the data
model.fit(X, y)

# make predictions on new data
# X_new = np.array([[2, 6, 4]])
X_new = np.array([[1.4]])
y_pred = model.predict(X_new)
print(y_pred)

Y = model.predict(X)
plt.scatter(X, y)
plt.scatter(X_new, y_pred, color='red')
plt.plot(X, Y, color='green')
plt.show()
