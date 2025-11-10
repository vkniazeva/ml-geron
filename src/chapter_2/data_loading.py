from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(points_number, is_test_split=True):
    mnist = fetch_openml('mnist_784',version=1)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.int8)
    if is_test_split:
        X_train, X_test, y_train, y_test = X[:points_number], X[points_number:], y[:points_number], y[points_number:]
        return X_train, X_test, y_train, y_test
    return X, y

def convert_to_image(dataset, datapoint_index):
    numpy_observation = dataset.to_numpy()[datapoint_index]
    image_dataset = numpy_observation.reshape(28, 28)
    return numpy_observation, image_dataset

def visualize_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()







