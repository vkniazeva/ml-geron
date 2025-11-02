from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate_rmse(model, X_set, y_set):
    predictions = model.predict(X_set)
    model_mse = mean_squared_error(y_set, predictions)
    model_rmse = np.sqrt(model_mse)
    print(f"Model RMSE on train set: {model_rmse:.2f}")


def evaluate_score(model, X_train, X_test, y_train, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Train score on train set: {train_score:.2f}")
    print(f"Test score on test set: {test_score:.2f}")

def evaluate_cross_validation(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
    model_rmse_scores = np.sqrt(-scores)
    print("-"*20, "K-Fold cross validation","-"*20)
    print("Sum of estimates: ", model_rmse_scores)
    print("Mean: ", model_rmse_scores.mean())
    print("Standard deviation: ", model_rmse_scores.std())