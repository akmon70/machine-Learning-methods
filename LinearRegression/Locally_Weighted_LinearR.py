import numpy as np
from numpy.linalg import pinv
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class LocallyWeightedRegression:
    """
    Locally weighted regression is a non-parametric algorithm that fits a 
    separate linear regression model for every test point by giving more 
    importance to the training points closer to the test point. 

    Parameters:
        tau (float): The bandwidth parameter which controls the amount of 
        weight given to each training example. Larger values of tau will result 
        in a smoother function but may cause the model to underfit.
    """

    def __init__(self, tau=0.1, normalize=True):
        self.tau = tau
        self.normalize = normalize
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
        X_norm = ((X - self.mean) / (self.std))
        return X_norm

    def kernel(self, query_point, X):
        """
        Calculates the weight for each training example based on its distance 
        from the query point using a Gaussian kernel function.

        Parameters:
            query_point (numpy array): A query point for which we want to 
            predict the output value.
            X (numpy array): The input training set.

        Returns:
            Weight_matrix (numpy matrix): A diagonal matrix of weights for each 
            training example.
        """
        n_samples, n_features = X.shape
        Weight_matrix = np.mat(np.eye(n_samples))
        for i in range(n_samples):
            # Calculate the Gaussian kernel for each training example
            Weight_matrix[i, i] = np.exp(
                ((X[i]-query_point) @ (X[i]-query_point).T)/(-2*self.tau*self.tau))
        return Weight_matrix

    def predict(self, X, Y, query_point):
        """
        Predicts the output value for a given query point by fitting a linear 
        regression model using a weighted least squares method.

        Parameters:
            X (numpy array): The input training set.
            Y (numpy array): The target output values for the training set.
            query_point (numpy array): A query point for which we want to 
            predict the output value.

        Returns:
            pred (float): The predicted output value for the query point.
        """
        q = np.mat([query_point, 1])
        X = np.hstack((X, np.ones((len(X), 1))))
        W = self.kernel(q, X)
        theta = pinv(X.T*(W*X))*(X.T*(W*Y))
        pred = q @ theta
        return pred

    def fit_and_predict(self, X, Y):
        """
        Fits a model for the input training set and returns the predicted 
        output values for the entire training set.

        Parameters:
            X (numpy array): The input training set.
            Y (numpy array): The target output values for the training set.

        Returns:
            Y_pred (numpy array): The predicted output values for the entire 
            training set.
        """
        if self.normalize:
            X = self.normalize_data(X)
        Y_pred, X_pred = [], np.linspace(-np.max(X), np.max(X), len(X))
        for x in X_pred:
            pred = self.predict(X, Y, x)
            Y_pred.append(pred[0][0])
        Y_pred = np.array(Y_pred)
        return Y_pred

    def score(self, Y, Y_pred):
        """
        Calculates the root mean square error (RMSE) for the predicted output 
        values and the actual output values.

        Parameters:
            Y (numpy array): The target output values for the training set.
            Y_pred (numpy array): The predicted output values for the training 
            set.

        Returns:
            rmse (float): The RMSE score.
        """
        return np.sqrt(np.mean((Y-Y_pred)**2))

    def fit_and_show(self, X, Y):
        # if self.normalize:
        #     X = self.normalize_data(X)
        Y_pred, X_pred = [], np.linspace(np.min(X), np.max(X), len(X))
        for x in X_pred:
            pred = self.predict(X, Y, x)
            Y_pred.append(pred[0][0])
        Y_pred = np.array(Y_pred)
        Y_pred = Y_pred.flatten()

        plt.style.use('seaborn')
        plt.title(
            "Prediction using Locally Weighted Linear Regression (LWLR) - tau = %.2f" % self.tau)
        plt.scatter(X, Y, color='red')
        # plt.scatter(X_pred, Y_pred, color='green')
        plt.plot(X_pred, Y_pred, color='blue')
        plt.show()


# ------------------------ test -----------------
# reading the csv files of the given dataset
dfx = pd.read_csv('weightedX.csv')
dfy = pd.read_csv('weightedY.csv')
# store the values of dataframes in numpy arrays
X = dfx.values
Y = dfy.values

model = LocallyWeightedRegression(tau=0.3)
Y_pred = model.fit_and_predict(X, Y)
model.fit_and_show(X, Y)
