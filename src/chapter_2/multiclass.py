import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_loading import load_data
from sklearn.svm import  LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import  OneVsOneClassifier


def build_svm(X_train, y_train, some_digit):
    svm_classifier = make_pipeline(
        StandardScaler(),
        LinearSVC(dual=True, max_iter=10000, random_state=42)
    )
    svm_classifier.fit(X_train, y_train)

    some_digit_predict = svm_classifier.predict(some_digit)
    print(f"Some digit prediction: {some_digit_predict}")

    some_digit_scores = svm_classifier.decision_function(some_digit)
    print("Some digit scores")
    print(some_digit_scores)

    max_index = np.argmax(some_digit_scores)
    print(f"The highest score's index: {max_index}")
    print(f"All classes: {svm_classifier.classes_}")
    print(f"Max score class by index = {max_index}: {svm_classifier.classes_[max_index]}")

def predicts_with_one_rest(X_train, y_train, some_digit):
    ovr_classifier = OneVsOneClassifier(LinearSVC(), n_jobs=-1)
    ovr_classifier.fit(X_train, y_train)
    predicted_value = ovr_classifier.predict(some_digit)
    print(f"OvO predicted value: {predicted_value}")
    print(f"OvO length: {len(ovr_classifier.estimators_)}")

def predict_with_sgd(X_train, y_train, some_digit):
    sgd_classifier = SGDClassifier()
    sgd_classifier.fit(X_train, y_train)
    predicted_value = sgd_classifier.predict(some_digit)
    print(f"SGD predicted value: {predicted_value}")
    print("SGD decision function outcome")
    print(sgd_classifier.decision_function(some_digit))

    print("Check cross validation results")
    cross_val_score_results = cross_val_score(sgd_classifier, X_train, y_train, cv=3, scoring="accuracy")
    print(np.round(cross_val_score_results, 3))


def main():
    X_train, X_test, y_train, y_test = load_data(2000)
    some_digit = X_train[15].reshape(1,-1)
    print(f"Some digit value: {y_train[15]}")
    print("--"*20, "One versus Rest (LinearSVC)", "--"*20)
    build_svm(X_train,y_train,some_digit)
    print("--" * 20, "One versus one (based on LinearSVC)", "--" * 20)
    predicts_with_one_rest(X_train,y_train,some_digit)
    print("--" * 20, "SGDClassifier", "--" * 20)
    predict_with_sgd(X_train,y_train,some_digit)


if __name__ == "__main__":
    main()