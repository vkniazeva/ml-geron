from scripts.chapter_1.data_loading import fetch_file_data
from scripts.chapter_1.data_splitting import prepare_train_test
from scripts.chapter_1.features.pipelines import build_full_pipeline
from scripts.chapter_1.model.train import *
from scripts.chapter_1.model.evaluate_model import evaluate_rmse, evaluate_score, evaluate_cross_validation, \
    evaluate_features_importance, evaluate_confidence_interval
from scripts.chapter_1.model.tune_model import *
from joblib import dump
from pathlib import Path

def run_workflow():
    housing = fetch_file_data("housing.csv", "chapter_1")
    X_train, X_test, y_train, y_test = prepare_train_test(housing)

    # file storing
    # base_dir = Path(__file__).resolve().parent
    # model_dir = base_dir / "model"
    # model_dir.mkdir(exist_ok=True)

    pipeline, num_features = build_full_pipeline(X_train)
    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared = pipeline.transform(X_test)

    # dump(pipeline, model_dir/"full_pipeline.joblib")

    print("=" * 25, "LinearRegression", "=" * 25)

    linear_model = train_linear_regression(X_train_prepared, y_train)
    evaluate_rmse(linear_model, X_train_prepared, y_train, "train")
    evaluate_score(linear_model, X_train_prepared, X_test_prepared, y_train, y_test)
    evaluate_cross_validation(linear_model, X_train_prepared, y_train)

    print("=" * 25, "DecisionTreeRegression", "=" * 25)

    tree_model = train_decision_tree_regression(X_train_prepared, y_train)
    evaluate_rmse(tree_model, X_train_prepared, y_train, "train")
    evaluate_score(tree_model, X_train_prepared, X_test_prepared, y_train, y_test)
    evaluate_cross_validation(tree_model, X_train_prepared, y_train)

    print("=" * 25, "Random Forest", "=" * 25)
    random_forest = train_random_forest(X_train_prepared, y_train)
    evaluate_rmse(random_forest, X_train_prepared, y_train, "train")
    evaluate_score(random_forest, X_train_prepared, X_test_prepared, y_train, y_test)
    evaluate_cross_validation(random_forest, X_train_prepared, y_train)

    grid_search = execute_grid_search(X_train_prepared, y_train)
    # random_search = execute_random_search(X_train_prepared, y_train)
    evaluate_features_importance(grid_search, pipeline, num_features)

    final_model = grid_search.best_estimator_
    final_prediction = final_model.predict(X_test_prepared)
    evaluate_rmse(final_model, X_test_prepared, y_test, "test")
    evaluate_confidence_interval(final_prediction, y_test)

if __name__ == "__main__":
    run_workflow()

