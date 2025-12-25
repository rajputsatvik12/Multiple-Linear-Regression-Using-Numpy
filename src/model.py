import numpy as np
from src.optimizer import GradientDescent


class LinearRegression:

    def __init__(self, learning_rate, n_iter=1000, regularisation_type=None, lambda_parameter=0.1):
        self.theta = np.array([0])
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.regularisation_type = regularisation_type
        self.lambda_parameter = lambda_parameter
        self.gradient_descent = GradientDescent(self.learning_rate, self.regularisation_type, self.lambda_parameter)

    @staticmethod
    def add_bias(X):
        ones = np.ones(shape=(len(X), 1))
        X_biased = np.concatenate((ones, X), axis=1)
        return X_biased

    def fit(self, X, y):
        X_biased = self.add_bias(X)
        self.theta = np.zeros(len(X_biased[0]))

        for i in range(self.n_iter):
            self.theta = self.gradient_descent.update(X_biased, y, self.predict(X), self.theta)

        return self.theta

    def predict(self, X):
        X_biased = self.add_bias(X)
        return X_biased @ self.theta

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2) / 2

    @staticmethod
    def r2score(y_true, y_pred):
        ssr = np.sum((y_true - y_pred) ** 2)
        sst = np.sum((y_true - np.mean(y_true)) ** 2)

        r2_score = 1 - ssr / sst
        return r2_score
