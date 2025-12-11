# Model tuning 

One of the logical steps after selecting a model (algorithm) is tuning the model to get better results by optimizing hyperparameters and improving its performance.

There are multiple ways to achieve this:

1. Randomly guessing hyperparameters
2. Grid Search
3. Randomized Search
4. Ensemble methods

## Grid Search

Grid Search (GridSearchCV) is an automated approach to check predefined combinations of hyperparameters.

1. Create a param_grid with all hyperparameter combinations
2. Only model parameters names from the class __init__ constructor are used (e.g., for RandomForestRegressor — "n_estimators", for KNeighborsRegressor — "n_neighbors").
3. GridSearch iterates through the dictionary containing {"param_name": [all values]} and checks every combination.
Example: {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]} -> 3 x 4 = 12 combinations
4. Each combination is evaluated using K-Fold cross-validation.
5. GridSearch stores: 
   * best_params_ (hyperparameters with the best mean score)
   * best_estimator_ (model trained on all data using best_params_)
   * cv_results_ (table with all combinations and their scores)

With GridSearchCV(refit=True), the model is retrained on the full dataset using the best parameters found during cross-validation.
If refit=False, the search results (best_params_) will still be available, but grid_search.best_estimator_ will not be trained 
and thus cannot be used for making predictions.

Key points:
1. No optimization
2. Time consuming
3. Each fit() inside cross-validation trains the model from scratch.

Results for RandomForest with GridSearch:

```
-------------------- Grid Search: best params --------------------
{'max_features': 8, 'n_estimators': 30}
-------------------- Grid Search: best estimators --------------------
RandomForestRegressor(max_features=8, n_estimators=30)
-------------------- Cross validation results --------------------
63930.00733095378 {'max_features': 2, 'n_estimators': 3}
55496.123750025276 {'max_features': 2, 'n_estimators': 10}
52721.50949936552 {'max_features': 2, 'n_estimators': 30}
....
```
The best parameters ({'max_features': 8, 'n_estimators': 30}) correspond to the upper boundary of the tested range.
This indicates that even higher values may potentially lead to a better-performing model.
The assumption is supported by the cross-validation results, where the MSE consistently 
decreases as both n_estimators and max_features increase.

For future model tuning, it makes sense to extend the search range for n_estimators, possibly testing significantly higher values 
(e.g., 50, 100, 200) to check where the performance plateaus.

## Randomized Search

Randomized Search works similarly to Grid Search but instead of checking all possible combinations from the parameter grid, it randomly samples parameter values from specified ranges or probability distributions.
The n_iter parameter controls the number of random combinations to evaluate.

How it works:
1. Define a dictionary of hyperparameters and their ranges or distributions 
2. RandomizedSearchCV generates n_iter random combinations of these hyperparameters. 
3. For each combination, the model is trained and evaluated on a validation fold, and each score is stored.
4. After all iterations are completed, Randomized Search provides:
    * best_params_ — the hyperparameters with the best mean score,
    * best_estimator_ — the trained model with those parameters,
    * cv_results_ — detailed statistics for all tried combinations

Key advantages:
* Faster than Grid Search (not all combinations are checked)
* Can explore wider parameter ranges

Results for RandomForest with RandomizedSearch:

```
-------------------- Randomized Search: best params --------------------
{'max_features': 6, 'n_estimators': 170}
-------------------- Randomized Search: best estimators --------------------
RandomForestRegressor(max_features=6, n_estimators=170)
-------------------- Cross validation results --------------------
49251.350429958 {'max_features': 8, 'n_estimators': 189}
50279.21012936821 {'max_features': 6, 'n_estimators': 24}
49462.21907806977 {'max_features': 4, 'n_estimators': 81}
....
```

According to the obtained results, the n_estimators parameter does not significantly improve the RMSE compared to lower values (e.g., 30), which indicates that the model has likely reached its performance plateau.
The best configuration was achieved with the following parameters: {'max_features': 6, 'n_estimators': 170}.
As a next step, it may be worth experimenting with other hyperparameters such as max_leaf_nodes or min_samples_split.
