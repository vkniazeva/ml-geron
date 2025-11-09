import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
from scipy import stats

def evaluate_rmse(model, X_set, y_set, set_type):
    predictions = model.predict(X_set)
    model_mse = mean_squared_error(y_set, predictions)
    model_rmse = np.sqrt(model_mse)
    print(f"\nModel RMSE on {set_type} set: {model_rmse:.2f}")


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

def evaluate_features_importance(search_model, full_pipeline, num_attributes):
    print("-"*20, "Feature importance array", "-"*20)
    feature_importances = search_model.best_estimator_.feature_importances_
    print(feature_importances)

    extra_attribs = ["rooms_per_hhold", "population_per_hhold", "bedrooms_per_hhold"]
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attributes + extra_attribs + cat_one_hot_attribs

    feature_importance_df = pd.DataFrame({
        "Feature": attributes,
        "Importance": feature_importances
    }).sort_values("Importance", ascending=False)

    print("\n" + "-"*20 + "Features importance with labels" + "-"*20 )
    print(feature_importance_df.to_string(index=False))

    # print("\nALTERNATIVE")
    # feature_names = full_pipeline.get_feature_names_out()
    # sorted_features = sorted(zip(feature_importances, feature_names), reverse=True)
    # feature_importance_df = pd.DataFrame(sorted_features, columns=["Importance", "Feature"])
    # print(feature_importance_df)

def evaluate_confidence_interval(predictions, target):
    confidence = 0.95
    squared_errors = (predictions - target) ** 2
    results = np.sqrt(stats.t.interval(confidence, len(squared_errors)-1,
                             loc=squared_errors.mean(), scale=stats.sem(squared_errors)))
    print("\nConfidence interval: ")
    print(results)

