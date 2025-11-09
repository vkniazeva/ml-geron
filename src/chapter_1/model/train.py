from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train_linear_regression(X_train, y_train):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    return linear_model


def train_decision_tree_regression(X_train, y_train):
    tree_model = DecisionTreeRegressor()
    tree_model.fit(X_train, y_train)
    return tree_model

def train_random_forest(X_train, y_train):
    random_forest = RandomForestRegressor()
    random_forest.fit(X_train, y_train)
    return random_forest




