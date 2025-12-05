import matplotlib.pyplot as plt
import numpy as np

from src.chapter_2.data_loading import *
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve
from sklearn.ensemble import RandomForestClassifier

def build_classifier():
    X_train, X_test, y_train, y_test = load_data(60000)

    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    sgd_classifier = SGDClassifier(random_state=42)

    sgd_classifier.fit(X_train, y_train_5)
    # print(sgd_classifier.predict([X_test[0]]))
    return sgd_classifier, X_train, X_test, y_train_5, y_test_5

def validate_cross_custom(model, X_train, y_train):
    skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_index, test_index in skfolds.split(X_train, y_train):
        clone_clf = clone(model)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train[train_index]
        X_test_folds = X_train[test_index]
        y_test_folds = y_train[test_index]
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_folds)
        n_correct = sum(y_pred == y_test_folds)
        print(n_correct/len(y_pred))

def validate_cross_library(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
    print(scores)

class Never5Classifier(BaseEstimator):
    def fit(self, X, y = None):
        pass
    def predict(self, X):
        return np.zeros((len(X)),dtype=bool)

# never_5_clf = Never5Classifier()

def calculate_model_accuracy(model, X_train, y_train):
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=3)
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_train, y_train_pred))

    print("PRECISION")
    print("{:.2f}".format(precision_score(y_train, y_train_pred)))

    print("RECALL")
    print("{:.2f}".format(recall_score(y_train, y_train_pred)))

    print("F1 score")
    print("{:.2f}".format(f1_score(y_train, y_train_pred)))


def make_decision(model, X_train):
    y_scores = model.decision_function([X_train[15]])
    print(f"INSTANCE SCORE")
    print(y_scores)
    threshold = 8000
    print(f"PREDICTION")
    y_digit_pred = (y_scores > threshold)
    print(y_digit_pred)

def build_trade_off_curve(model, X_train, y_train):
    y_scores = cross_val_predict(model, X_train, y_train, cv=3, method="decision_function")
    precision, recall, thresholds = precision_recall_curve(y_train, y_scores)
    prec = precision[:-1]
    print("90% precision threshold")
    threshold_90_precision = thresholds[np.argmax(prec >= 0.90)]
    print(threshold_90_precision)
    # plt.plot(thresholds, precision[:-1],"b--", label="precision")
    # plt.plot(thresholds, recall[:-1], "g--", label="recall")
    # plt.xlabel("Threshold")
    # plt.ylabel("Score")
    # plt.title("Precision/Recall Trade-off")
    # plt.legend()
    # plt.grid(True)
    # plt.ylim([0, 1])
    # plt.show()
    return y_scores, threshold_90_precision

def build_custom_predictions(y_train, scores, threshold):
    y_train_pred = (scores >= threshold)
    print("PRECISION WITH A NEW THRESHOLD")
    print("{:.3f}".format(precision_score(y_train, y_train_pred)))
    print("RECALL WITH A NEW THRESHOLD")
    print("{:.3f}".format(recall_score(y_train, y_train_pred)))

def build_roc_curve(y_train, scores, label):
    fpr, tpr, threshold = roc_curve(y_train, scores)
    # plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--', label="Random")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC curve")
    plt.grid(True)
    plt.legend()

    auc = roc_auc_score(y_train, scores)
    print("AREA UNDER CURVE")
    print("{:.3f}".format(auc))

def build_random_forest(X_train, y_train):
    forest_classifier = RandomForestClassifier(random_state=42)
    y_probas_forest = cross_val_predict(forest_classifier, X_train, y_train, cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1]
    return y_scores_forest, forest_classifier

def main():
    model, X_train, X_test, y_train_5, y_test_5 = build_classifier()
    print("CUSTOM CROSS VALIDATION")
    # validate_cross_custom(model, X_train, y_train_5)
    # print("BUILT IN CROSS VALIDATION")
    # validate_cross_library(model, X_train, y_train_5)
    # print("MODEL ACCURACY METRICS")
    # calculate_model_accuracy(model, X_train, y_train_5)
    # make_decision(model, X_train)
    scores, threshold = build_trade_off_curve(model, X_train, y_train_5)
    # print("90% PRECISION THRESHOLD")
    # build_custom_predictions(y_train_5, scores, threshold)
    build_roc_curve(y_train_5, scores, "SGD")
    scores_forest, forest_classifier = build_random_forest(X_train, y_train_5)
    build_roc_curve(y_train_5, scores_forest, "Random Forest")
    plt.show()
    calculate_model_accuracy(forest_classifier, X_train, y_train_5)



if __name__ == "__main__":
    main()






