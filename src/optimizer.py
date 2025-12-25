import numpy as np


class GradientDescent:

    def __init__(self, learning_rate, regularisation_type, lambda_parameter):
        self.learning_rate = learning_rate
        self.regularisation_type = regularisation_type
        self.lambda_parameter = lambda_parameter

    def update(self, X, y_true, y_pred, theta):
        error = (y_pred - y_true) / len(y_true)
        gradient = X.T @ error / len(X)

        if self.regularisation_type == 'L1':
            theta_reg = theta.copy()
            theta_reg[0] = 0

            gradient += self.lambda_parameter / len(y_true) * np.sign(theta_reg)
        elif self.regularisation_type == 'L2':
            theta_reg = theta.copy()
            theta_reg[0] = 0

            gradient += theta_reg * self.lambda_parameter / len(y_true)

        return theta - self.learning_rate * gradient
