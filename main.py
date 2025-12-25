from src.model import LinearRegression
from src.utils import load_data, train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as sklearnLR
from sklearn.metrics import mean_squared_error, r2_score


def main():
    filepath = r'C:\Users\Dell\PycharmProjects\LinearRegression\data\sample_data.csv'
    X, y = load_data(filepath)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.3)
    CustomLR = LinearRegression(learning_rate=0.05, n_iter=10000)
    SKLR = sklearnLR()

    CustomLR.fit(X_train, y_train)
    print("Custom Model:")
    print("Learned intercept: ", CustomLR.theta[0])
    print("Learned slope: ", CustomLR.theta[1])
    y_train_pred_custom = CustomLR.predict(X_train)
    y_test_pred_custom = CustomLR.predict(X_test)

    print("Training metrics:")
    print(CustomLR.mean_squared_error(y_train, y_train_pred_custom))
    print(CustomLR.r2score(y_train, y_train_pred_custom))

    print("Evaluation metrics:")
    print(CustomLR.mean_squared_error(y_test, y_test_pred_custom))
    print(CustomLR.r2score(y_test, y_test_pred_custom))
    print()
    SKLR.fit(X_train, y_train)
    print("Sklearn Model:")
    print(SKLR.coef_)

    y_train_pred_sklearn = SKLR.predict(X_train)
    y_test_pred_sklearn = SKLR.predict(X_test)

    print("Training metrics:")
    print(mean_squared_error(y_train, y_train_pred_sklearn))
    print(r2_score(y_train, y_train_pred_sklearn))

    print("Evaluation metrics:")
    print(mean_squared_error(y_test, y_test_pred_sklearn))
    print(r2_score(y_test, y_test_pred_sklearn))

    plt.figure()
    plt.title("Custom Model vs Sklearn Model")
    plt.scatter(X, y)
    y_pred_custom = CustomLR.predict(X)
    y_pred_sklearn = SKLR.predict(X)
    plt.plot(X, y_pred_custom, color='Red', label='Custom Model')
    plt.plot(X, y_pred_sklearn, color='Blue', label='Sklearn Model')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
