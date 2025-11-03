import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def execute_grid_search(X_train, y_train):
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)

    grid_search.fit(X_train, y_train)
    print_search_results("Grid Search",grid_search)


def execute_random_search(X_train, y_train):
    param_distribs = {
        'n_estimators': randint(low=10, high=200),
        'max_features': randint(low=2, high=10),
    }

    forest_reg = RandomForestRegressor()
    random_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,n_iter=20, cv=5,
                                       scoring='neg_mean_squared_error', random_state=42, return_train_score=True)
    random_search.fit(X_train, y_train)
    print_search_results("Randomized Search", random_search)


def print_search_results(search_name,search_model):
    print(f'\n{"-" * 20} {search_name}: best params {"-" * 20}')
    print(search_model.best_params_)
    print(f'{"-" * 20} {search_name}: best estimator {"-" * 20}')
    print(search_model.best_estimator_)
    print(f'{"-" * 20} Cross validation results {"-" * 20}')

    cvres = search_model.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(f"{np.sqrt(-mean_score):.2f}", params)





