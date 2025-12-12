import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from data_loading import load_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict


def search_params(X_train, y_train, knn):
    param_grid = [
        {'n_neighbors': range(1, 30),
         'weights': ['uniform', 'distance']}
    ]
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid,
                               scoring='accuracy', return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best combination: {grid_search.best_params_}")
    print(f"Best scores (accuracy): {grid_search.best_score_:.4f}")
    return grid_search.best_params_


def shift_pixels_left(X_train):
    X_train_shifted = []
    for img in X_train:
        img = img.reshape(28, 28)
        shifted = np.zeros_like(img)
        shifted[:, :-1] = img[:, 1:]
        X_train_shifted.append(shifted.reshape(-1))
    return np.array(X_train_shifted)

def shift_pixels_right(X_train):
    X_train_shifted = []
    for img in X_train:
        img = img.reshape(28, 28)
        shifted = np.zeros_like(img)
        shifted[:, 1:] = img[:, :-1]
        X_train_shifted.append(shifted.reshape(-1))
    return np.array(X_train_shifted)

def shift_pixels_up(X_train):
    X_train_shifted = []
    for img in X_train:
        img = img.reshape(28, 28)
        shifted = np.zeros_like(img)
        shifted[:-1, :] = img[1:, :]
        X_train_shifted.append(shifted.reshape(-1))
    return np.array(X_train_shifted)

def shift_pixels_down(X_train):
    X_train_shifted = []
    for img in X_train:
        img = img.reshape(28, 28)
        shifted = np.zeros_like(img)
        shifted[1:, :] = img[:-1, :]
        X_train_shifted.append(shifted.reshape(-1))
    return np.array(X_train_shifted)


def main():
    # loading the data
    X_train, X_test, y_train, y_test = load_data(2000, is_test_split=True)

    # shifted arrays
    X_shift_left = shift_pixels_left(X_train)
    X_shift_right = shift_pixels_right(X_train)
    X_shift_up = shift_pixels_up(X_train)
    X_shift_down = shift_pixels_down(X_train)

    # joining all shifted arrays by columns
    X_train_aug = np.vstack([X_train,
                             X_shift_left,
                             X_shift_right,
                             X_shift_up,
                             X_shift_down])

    y_train_aug = np.hstack([y_train,
                             y_train,
                             y_train,
                             y_train,
                             y_train])

    # scaling for KNN - important because of the distance
    scaler = MinMaxScaler()
    X_train_prepared = scaler.fit_transform(X_train_aug)
    # create a model
    knn_classifier = KNeighborsClassifier()
    params = search_params(X_train_prepared, y_train_aug, knn_classifier)

    #trainging the model
    n_neighbors = params['n_neighbors']
    weights = params['weights']
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    scores = cross_val_score(knn_classifier, X_train_prepared, y_train_aug)
    print(f"Best model accuracy: {np.round(scores, 2)}")




if __name__ == "__main__":
    main()