import numpy as np


class LocalWeightedLinearRegression:
    def __init__(self, tau=0.5, alpha=0.1, num_iterations=1000):
        """
        Local Weighted Linear Regression with Gradient Descent
        tau: bandwidth parameter for local weighting
        alpha: learning rate for gradient descent
        num_iterations: number of iterations for gradient descent
        """
        self.tau = tau
        self.alpha = alpha
        self.num_iterations = num_iterations

    def fit(self, X, y):
        """
        Fit the model to the training data
        X: input features (m x n matrix)
        y: output labels (m x 1 vector)
        """
        # Add a column of ones to X to account for the bias term
        m, n = X.shape
        X = np.hstack((np.ones((m, 1)), X))

        # Initialize the weights to 1
        weights = np.ones((m, 1))

        # Initialize the parameters to 0
        theta = np.zeros((n + 1, 1))

        # Loop over the number of iterations
        for i in range(self.num_iterations):
            # Compute the predicted values
            y_pred = X.dot(theta)

            # Compute the errors
            errors = y - y_pred

            # Compute the weighted errors
            weighted_errors = weights * errors

            # Compute the diagonal weight matrix
            W = np.diag(
                np.exp(-((X[:, 1:] - X[:, 1:].T)**2).sum(axis=1)/(2*self.tau**2)))

            # Compute the gradient
            gradient = X.T.dot(W).dot(X).dot(theta) - \
                X.T.dot(W).dot(y) + self.alpha * theta

            # Update the weights
            weights = np.diag(
                np.exp(-((X[:, 1:] - X[:, 1:].T)**2).sum(axis=1)/(2*self.tau**2)))

            # Update the parameters using the gradient descent update rule
            theta = theta - self.alpha * gradient

        self.theta = theta

    def predict(self, X):
        """
        Predict the output labels for new input features
        X: input features (m x n matrix)
        """
        # Add a column of ones to X to account for the bias term
        m, n = X.shape
        X = np.hstack((np.ones((m, 1)), X))

        # Compute the predicted values using the learned parameters
        y_pred = X.dot(self.theta)

        return y_pred
