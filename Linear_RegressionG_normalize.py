
import numpy as np


class LinearRegression:
    def __init__(self, alpha=0.01, n_iterations=1000, fit_intercept=True, normalize=True):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.coefficients = None
        self.mean = None
        self.std = None

    def normalize_data(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_normalized = (X - self.mean) / self.std
        return X_normalized

    def add_intercept(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def compute_cost(self, X, y, coefficients):
        n_samples = X.shape[0]
        y_pred = X @ coefficients
        mse = np.sum((y_pred - y) ** 2) / (2 * n_samples)
        return mse

    def gradient_descent_step(self, X, y, coefficients):
        n_samples = X.shape[0]
        y_pred = X @ coefficients
        error = y_pred - y
        gradient = (X.T @ error) / n_samples
        coefficients = coefficients - self.alpha * gradient
        return coefficients

    def fit(self, X, y):
        if self.normalize:
            X = self.normalize_data(X)
        if self.fit_intercept:
            X = self.add_intercept(X)
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features)
        for i in range(self.n_iterations):
            self.coefficients = self.gradient_descent_step(
                X, y, self.coefficients)

    def predict(self, X):
        if self.normalize:
            X = self.normalize_data(X)
        if self.fit_intercept:
            X = self.add_intercept(X)
        return X @ self.coefficients


# -------------------- test -------------
