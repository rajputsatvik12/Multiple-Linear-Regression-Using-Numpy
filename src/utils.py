import numpy as np


def load_data(filepath):
    data = np.genfromtxt(filepath,delimiter=',',skip_header=True)
    # input features : entire dataset except the last column
    X = data[:, :-1]
    # target variable : last column
    y = data[:, -1]

    return X, y


def train_test_split(X, y, test_size):
    split_idx = np.round((1 - test_size) * len(X))
    split_idx = int(split_idx)

    # 1st split_idx examples will be used for training and rest for evaluating the model
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    return X_train, y_train, X_test, y_test
