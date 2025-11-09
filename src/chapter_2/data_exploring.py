from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784',version=1)
print(mnist.keys())
print(mnist.data)

X, y = mnist["data"], mnist["target"]
y = y.astype(np.int8)
print(X.shape)
print(y.shape)

some_digit = X.to_numpy()[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X[:60000], X[60000:],
                              y[:60000], y[60000:])


