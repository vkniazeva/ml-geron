import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_loading import load_data
from sklearn.svm import  LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import  OneVsOneClassifier


def build_svm(X_train, y_train, some_digit):
    svm_classifier = LinearSVC(dual=True, max_iter=10000, random_state=42)
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
    return sgd_classifier

def analyse_errors(model, X_train, y_train):
    y_train_pred = cross_val_predict(model, X_train, y_train)
    conf_matrix = confusion_matrix(y_train, y_train_pred)
    print("Confusion matrix")
    print(conf_matrix)
    plt.matshow(conf_matrix, cmap=plt.cm.gray)

    rows_sums = conf_matrix.sum(axis=1, keepdims=True)
    norm_conf_matrix = conf_matrix / rows_sums

    np.fill_diagonal(norm_conf_matrix, 0)
    plt.matshow(norm_conf_matrix, cmap=plt.cm.gray)

    cl_a, cl_b = 3, 5

    X_aa = X_train[ (y_train == cl_a) & (y_train_pred == cl_a)]
    X_ab = X_train[ (y_train == cl_a) & (y_train_pred == cl_b)]
    X_ba = X_train[ (y_train == cl_b) & (y_train_pred == cl_a)]
    X_bb = X_train[ (y_train == cl_b) & (y_train_pred == cl_b)]

    simple_plot_digits(X_aa[:25], "3s → 3s (correct)")
    simple_plot_digits(X_ab[:25], "3s → 5s (incorrect)")
    simple_plot_digits(X_ba[:25], "5s → 3s (incorrect)")
    simple_plot_digits(X_bb[:25], "5s → 5s (correct)")
    plt.show()

def simple_plot_digits(images, title):
    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].reshape(28, 28), cmap='binary')
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()

def main():
    X_train, X_test, y_train, y_test = load_data(2000)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    some_digit = X_train[15].reshape(1,-1)
    # print(f"Some digit value: {y_train[15]}")
    # print("--"*20, "One versus Rest (LinearSVC)", "--"*20)
    # build_svm(X_train_scaled,y_train,some_digit)
    # print("--" * 20, "One versus one (based on LinearSVC)", "--" * 20)
    # predicts_with_one_rest(X_train,y_train,some_digit)
    print("--" * 20, "SGDClassifier", "--" * 20)
    model = predict_with_sgd(X_train,y_train,some_digit)
    analyse_errors(model, X_train_scaled, y_train)


if __name__ == "__main__":
    main()