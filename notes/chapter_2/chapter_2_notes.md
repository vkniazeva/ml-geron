# Classification

Scikit-Learn datasets structure:
- 'DESCR' - dataset description
- 'data' - data points array (DataFrame)
- 'target' - array with labels (Series)


## Learning algorithm

Complementing classical machine learning gradient descent algorithms, scikit-learn provides a stochastic gradient descent (SGD) learning approach.
Compared to classical gradient descent methods, SGD (available in the linear_model module) computes the error between the predicted and actual value not over the entire dataset, but only on a small batch (a subset of samples).

In essence, the algorithm randomly selects one or several data points, makes predictions for them, and updates the model’s weights based on the resulting error.
Repeating this process many times allows the model to update weights faster and more efficiently.
However, the downside of this approach is that the optimization path becomes noisier, and the algorithm might miss the exact optimal weights.

There are several common variants of gradient descent:

| Approach       | How it works   | Remarks                                 |
|----------------|----------------|-----------------------------------------|
| Batch GD       | Entire dataset | Accurate but computationally expensive  |
| Stochastic GD  | Single sample  | Fast but noisy and less stable          |
| Mini-Batch GD  | Small subset   | A trade-off between speed and accuracy  |


## Useful Python insights

- to_numpy() - conversing an object to a np.ndarray. If from DataFrame, column names are removed
- reshape() - taking np.ndarray array and changing it to a 2D array of a defined size
- imshow() - creating an image, cmap - colormap ("binary" - black and white)

