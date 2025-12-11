# Building model

As a part of a data processing pipeline, the following steps were taken:
1. Loading the dataset
2. Splitting the data into training and test sets
3. Introducing a custom transformer for a feature combination
4. Setting up a numeric data processing pipeline:
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

To compare both models' performance more accurately, K-fold cross-validation was used, showing the following results:

Linear model:
```
Sum of estimates:  [71762.76364394 64114.99166359 67771.17124356 68635.19072082
 66846.14089488 72528.03725385 73997.08050233 68802.33629334
 66443.28836884 70139.79923956]
Mean:  69104.07998247063
Standard deviation:  2880.328209818061
```

DecisionTree:
```
Sum of estimates:  [71358.62170279 70242.08871925 67819.48482041 71352.69259929
 69868.04918266 78353.60935234 71019.00058617 73202.51635956
 68335.24130461 69159.04378436]
Mean:  71071.03484114332
Standard deviation:  2856.414353949065
```

DecisionTree as visible from the scores performs even worse than LinearRegression, but in general 
both models fail to properly make predictions.

As the Decision Tree model showed, it performs quite well on the training set, but its generalization 
capability is much worse even compared to Linear Regression. However, it can be assumed that the relationships between 
variables are mostly non-linear. Therefore, an Ensemble Learning method such as Random Forest can be tried, as it builds 
multiple decision trees on random subsets of data and features, then aggregates their predictions.

For the random forest, the following results were obtained:

```
Model RMSE on train set: 18654.64
Train score on train set: 0.97
Test score on test set: 0.82
-------------------- K-Fold cross validation --------------------
Sum of estimates:  [51258.25775776 48851.19565398 46810.28701105 51953.63411969
 47137.58624757 51590.20061796 52489.0337876  49805.00771232
 48482.69530175 53865.32837827]
Mean:  50224.32265879509
Standard deviation:  2249.2032230580016
```

As the metrics show, the mean RMSE from K-Fold cross-validation is significantly higher 
than on the training set overall, which clearly indicates overfitting.

As a further step before tuning the model itself, an ML algorithm should be selected 
by trying out models from different algorithmic families. 
Only the top 2–5 most promising algorithms should be selected for final tuning.

## K-fold cross validation
K-fold cross-validation stands for a model evaluation approach. It splits a training set to a 
specified subsets number (folds) and iterates through the entire training set by holding one of 
 the folds on each iteration. Such a solution allows evaluating the model performance in a more 
robust way.

How does it work?
```
Training set: [0 1 2 3 4 5 6 7 8 9]
K-fold = 5

1st iteration:
Train: [2 3 4 5 6 7 8 9] → [2,3,4,5,6,7,8,9]
Test:  [0 1] → [0,1]

2nd iteration:
Train: [0 1 4 5 6 7 8 9] → [0,1,4,5,6,7,8,9]
Test:  [2 3] → [2,3]
...
```
KFold is typically used for regression tasks, while for classification problems,
StratifiedKFold is preferred because it preserves the class distribution of the target variable.

The score() method of KFold/StratifiedKFold sequentially calls model.fit() on each training subset (fold) along with its corresponding labels.
After predict() method is executed with the fold values.
Finally, scoring is applied. For regression, the common metrics are defined as follows (note: higher values indicate better performance):
* neg_mean_squared_error -> (-MSE)
* neg_root_mean_squared_error -> (-SQRT)
* r2 - determination coef
* neg_mean_absolute_error -> (-MAE)

As a result, the scoring process returns a metric for each iteration, after which mean() and std() can be applied to compute the average performance and its variability.


