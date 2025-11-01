# Building model

As a part of a data processing pipeline, the following steps were taken:
1. Loading the dataset
2. Splitting the data into training and test sets
3. Introducing a custom transformer for a feature combination
4. eSetting up a numeric data processing pipeline:
   * replacing missing values with the median
   * applying the custom feature transformer (point 3)
   * standardizing the data
5. Calling fit_transform() on the training set
6. Transforming the test set using the already fitted pipeline
7. Trying out different models
8. Evaluating them

## Applying ML 

A typical first step in building a predictive model is to start with a linear regression.
Despite its simplicity, a linear model can perform surprisingly well on small or medium-sized 
datasets where the relationships between features and the target variable are almost linear.

In this case, after computing the Root Mean Squared Error (RMSE) on the training set, 
it turned out to be quite high considering the target variable’s range
— suggesting possible underfitting.
To verify this, the score() method was used to compare the R² (coefficient of determination) 
between the training and test sets. (predicted values deviation vs deviation from mean)


Linear Model results:
```
Model RMSE on train set: 68627.87
Train score on train set: 0.65
Test score on test set: 0.66
```
Such low R² values for both the training and test sets indicate underfitting — the model is too simple 
to capture the underlying patterns in the data.
Since no regularization was applied here, underfitting seems to be the main reason for poor performance.
This suggests that Linear Regression was too simple to model potential non-linear relationships, 
and trying a more flexible model makes sense.

A Decision Tree Regressor was used next.
```
Model RMSE on train set: 0.00
Train score on train set: 1.00
Test score on test set: 0.61
```
As is typical for Decision Trees, this is a clear case of overfitting — the model fits the training data perfectly 
but fails to generalize to unseen data (the test R² is much lower).

