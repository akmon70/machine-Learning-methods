
import numpy as np
import matplotlib.pyplot as plt


class LocallyWeightedLinearRegression:
    """
    Locally weighted linear regression model using gradient descent with normalization.

    Parameters
    ----------
    learning_rate : float, default=0.1
        The learning rate to use in the gradient descent algorithm.
    max_iterations : int, default=1000
        The number of iterations to use in the gradient descent algorithm.
    normalize : bool, default=True
        Whether to normalize the input data.
    tau : float, default=1.1
        The bandwidth parameter for the weighted least squares algorithm.

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

    def __init__(self, tau=0.1, normalize=True, learning_rate=0.1, max_iterations=1000, tol=0.0001):
        self.tau = tau
        self.normalize = normalize
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol
        self.mean = None
        self.std = None
        self.theta = None

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

    def weight_matrix(self, X):
        """
        This method computes the weight matrix for the input data X. The weight matrix is a square matrix where the (i,j)th element represents the weight of the jth point with respect to the ith point. The weight is computed using a Gaussian kernel function with bandwidth parameter tau.
        """
        n_samples = X.shape[0]
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            xi = X[i]
            diff = X - xi
            diff_norm = np.linalg.norm(diff, axis=1)
            w = np.exp(-diff_norm / (2 * self.tau**2))
            W[i, :] = w / np.sum(w)
        return W

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
        self.theta = np.zeros((n_features + 1, 1))
        X = np.hstack((np.ones((n_samples, 1)), X))
        W = self.weight_matrix(X[:, 1:])
        y = y.reshape(-1, 1)
        for i in range(self.max_iterations):
            h = X @ self.theta
            errors = y - h
            gradient = (X.T @ (W @ errors)) / n_samples
            self.theta += self.learning_rate * gradient
            if np.max(np.abs(gradient)) < self.tol:
                break

    def predict(self, X):
        if self.normalize:
            X = (X - self.mean) / self.std
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred = X @ self.theta
        return y_pred


# -------------------------- test ---------------
# create some sample data
# X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 7, 9]])
# y = np.array([6, 15, 24, 7, 100])
X = 100 * np.random.rand(100, 1)
y = 40 + 30 * X + 400 * np.random.randn(100, 1)

# create a LinearRegressionGD instance
model = LocallyWeightedLinearRegression(
    learning_rate=0.1, max_iterations=1000, normalize=True)

# fit the model to the data
model.fit(X, y)

# make predictions on new data
# X_new = np.array([[2, 6, 4]])
X_new = np.array([[5]])
y_pred = model.predict(X_new)
print(y_pred)

Y = model.predict(X)
plt.scatter(X, y)
plt.scatter(X_new, y_pred, color='red')
plt.plot(X, Y, color='green')
plt.show()
