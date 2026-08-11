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

