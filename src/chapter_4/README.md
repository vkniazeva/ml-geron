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

## Stochastic Gradient descent

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


## Batch Gradient Descent

- Batch Gradient Descent: uses all examples to calculate gradient, updates weights once per epoch (stable, slow)
- Stochastic Gradient Descent:  uses one example at a time, updates weights immediately (fast, noisy)




