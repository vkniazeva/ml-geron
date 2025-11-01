from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train_linear_regression(X_train, y_train):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    return linear_model


def train_decision_tree_regression(X_train, y_train):
    tree_model = DecisionTreeRegressor()
    tree_model.fit(X_train, y_train)
    return tree_model


def evaluate_rmse(model, X_test, y_test):
    predictions = model.predict(X_test)
    model_mse = mean_squared_error(y_test, predictions)
    model_rmse = np.sqrt(model_mse)
    print(f"Model RMSE on train set: {model_rmse:.2f}")


def evaluate_score(model, X_train, X_test, y_train, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Train score on train set: {train_score:.2f}")
    print(f"Test score on test set: {test_score:.2f}")
