from scripts.chapter_1.data_loading import fetch_file_data
from scripts.chapter_1.data_splitting import prepare_train_test
from scripts.chapter_1.features.pipelines import build_full_pipeline
from scripts.chapter_1.model.train import *
from scripts.chapter_1.model.evaluate_model import evaluate_rmse, evaluate_score, evaluate_cross_validation
from joblib import dump
from pathlib import Path

def run_workflow():
    housing = fetch_file_data("housing.csv", "chapter_1")
    X_train, X_test, y_train, y_test = prepare_train_test(housing)

    # file storing
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "model"
    model_dir.mkdir(exist_ok=True)


    # building pipeline

    pipeline = build_full_pipeline(X_train)
    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared = pipeline.transform(X_test)

    # storing pipeline
    dump(pipeline, model_dir/"full_pipeline.joblib")

    print("=" * 25, "LinearRegression", "=" * 25)

    linear_model = train_linear_regression(X_train_prepared, y_train)
    evaluate_rmse(linear_model, X_train_prepared, y_train)
    evaluate_score(linear_model, X_train_prepared, X_test_prepared, y_train, y_test)
    evaluate_cross_validation(linear_model, X_train_prepared, y_train)

    # storing linear model
    dump(linear_model, model_dir/"linear_model.joblib")

    print("=" * 25, "DecisionTreeRegression", "=" * 25)

    tree_model = train_decision_tree_regression(X_train_prepared, y_train)
    evaluate_rmse(tree_model, X_train_prepared, y_train)
    evaluate_score(tree_model, X_train_prepared, X_test_prepared, y_train, y_test)
    evaluate_cross_validation(tree_model, X_train_prepared, y_train)

    # storing decision tree model
    dump(tree_model, model_dir/"tree_model.joblib")

    print("=" * 25, "Random Forest", "=" * 25)
    random_forest = train_random_forest(X_train_prepared, y_train)
    evaluate_rmse(random_forest, X_train_prepared, y_train)
    evaluate_score(random_forest, X_train_prepared, X_test_prepared, y_train, y_test)
    evaluate_cross_validation(random_forest, X_train_prepared, y_train)

    # storing random forest model
    dump(random_forest, model_dir/"random_forest.joblib")


if __name__ == "__main__":
    run_workflow()

