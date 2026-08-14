# Linear Regression

Linear regression is a class of models that predicts the numerical value (y) as a linear combination of input features (x),

In general, it can be described mathematically:

y_pred = w_0 + w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n

where:
- y_pred - a model's prediction
- w_{} - model's params/weights
- x_{} - input features

## When linear regression will work?

1. Linearity between the features and the target (appr.)
2. No strong multicollinearity between the features 
3. Results interpretability requirements
4. Small-medium datasets
5. Fast baseline

### When linear regression won't work

1. Data is not linear (but polynomials can be added)
2. Tons of outliers
3. High accuracy with the complex dependencies

## Model's optimization

| Error | Name                        | When to use                                                 |
|-------|-----------------------------|-------------------------------------------------------------|
| MSE   | Mean Square Error           | Standard dataset, no outliers, simple gradient computation  |
| MAE   | Mean Absolute Error         | Tons of outliers, interpretability, no differentiation in 0 |
| RMSE  | Root Mean Squared Error     | Outliers intolerant, better interpretability (vs MSE)       |
| Huber | Balance between MSE and MAE | Outliers robust, differentiation                            |

Huber works like a threshold error computation switcher: on the small errors (below the threshold) it behaves like MSE,
and on the larger errors as MAE. So model optimization does not assign huge penalties to the big error values.

## Gradient descent

Gradient descent is the most common optimization algorithm, capable of finding an optimal solution for a wide range of tasks. 
The core idea lies in the iterative adaptation of parameters to reach the minimum of a given cost function. 
Gradient descent computes the local gradient (a vector of partial derivatives) and updates the parameters in the direction of the steepest decrease.
When gradient = 0 -> the minimum is found.

The learning rate essentially controls how big of a step we take to reach the minimum of the cost function.

The MSE cost function for linear regression is convex, which provides the following advantages:
- has a single global minimum (no local minima)
- continuous and differentiable everywhere
- the gradient decreases monotonically

> **_NOTE:_** feature scaling is essential for a minimum search


## Types of Gradient Descent

- Batch Gradient Descent: uses all examples to calculate gradient, updates weights once per epoch (stable, slow)
- Stochastic Gradient Descent:  uses one example at a time, updates weights immediately (fast, noisy)

In practice, SGD achieves better results with irregular cost functions.
Due to its stochastic nature, SGD has a higher probability of discovering more optimal weights. 
Its movements are non-deterministic and allow it to  escape local minima. 
BGD, however, typically becomes trapped when it encounters a local minimum.

Shuffling is crucial for SGD because it requires samples to be independent and identically distributed (IID). 
Without shuffling, ordered data introduces bias in gradient estimates. 
For example, if positive and negative samples are grouped together, the gradient will be skewed in one direction,
leading to unstable convergence and potentially suboptimal solutions.

## Learning rate

Learning rate adjustment improves the model's training process. 
A common approach is to start with a higher learning rate, allowing optimization algorithms to take larger steps toward the minimum. 
However, as the algorithm approaches the minimum, the gradient becomes smaller, resulting in naturally smaller step sizes. 
To ensure precise convergence and avoid overshooting the global minimum, the learning rate should be further decreased in later stages of training.

'Learning schedule' stands for a function which determines the learning rate on every iteration.






