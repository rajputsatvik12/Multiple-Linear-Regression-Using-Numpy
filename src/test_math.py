import numpy as np
from model import LinearRegression


def test_mse():
    """Sample data with MSE = 0 to confirm that the class method works as expected"""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    mse = LinearRegression.mean_squared_error(y_true, y_pred)
    if mse == 0:
        print("MSE Test Passed!")
    else:
        print("MSE Test Failed!")


def test_dimensions():
    """To check whether the matrices in the project have expected dimensions"""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([6, 7, 8, 9, 10])

    model = LinearRegression(learning_rate=0.1, n_iter=100)
    X_biased = model.add_bias(X)
    if X_biased.shape == (5, 2):
        print("Bias added successfully!")
    else:
        print("Adding bias Failed!")

    model.fit(X, y)
    if model.theta.shape == (2, ):
        print("Learning successful, no shape mismatch!")
    else:
        print("Shape mismatch in parameter array!")


def test_model():
    model = LinearRegression(learning_rate=0.1)
    X = np.array([[1], [2], [3], [4], [5]])
    y = 5 + 2 * X.flatten()
    # known output : bias term should be close to 5 and slope should be close to 2
    model.fit(X, y)
    if np.isclose(model.theta[0], 5, rtol=0.5) and np.isclose(model.theta[1], 2,rtol=0.1):
        print("Model fits perfectly!")
    else:
        print("Incorrect fit!")

if __name__ == "__main__":
    test_mse()
    test_model()
    test_dimensions()