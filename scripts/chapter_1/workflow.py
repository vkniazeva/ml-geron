from scripts.chapter_1.data_loading import fetch_file_data
from scripts.chapter_1.data_splitting import prepare_train_test
from scripts.chapter_1.features.pipelines import build_full_pipeline
from scripts.chapter_1.model.train import train_linear_regression, train_decision_tree_regression, evaluate_rmse, evaluate_score

def run_workflow():
    housing = fetch_file_data("housing.csv", "chapter_1")
    X_train, X_test, y_train, y_test = prepare_train_test(housing)

    pipeline = build_full_pipeline(X_train)
    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared = pipeline.transform(X_test)

    print("=" * 25, "LinearRegression", "=" * 25)

    linear_model = train_linear_regression(X_train_prepared, y_train)
    evaluate_rmse(linear_model, X_train_prepared, y_train)
    evaluate_score(linear_model, X_train_prepared, X_test_prepared, y_train, y_test)

    print("=" * 25, "DecisionTreeRegression", "=" * 25)

    tree_model = train_decision_tree_regression(X_train_prepared, y_train)
    evaluate_rmse(tree_model, X_train_prepared, y_train)
    evaluate_score(tree_model, X_train_prepared, X_test_prepared, y_train, y_test)


if __name__ == "__main__":
    run_workflow()

