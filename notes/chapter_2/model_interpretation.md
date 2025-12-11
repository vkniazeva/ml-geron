# Model interpretation

After completing the model tuning step, when the final model is obtained,
it’s important to evaluate and interpret its behavior.

## Features importance 
Compared to the initial dataset, which included 12 features, the final model 
now uses 16 features after all data transformations.
Each of these features can be evaluated based on its importance for the model.

The feature_importances_ attribute of the trained model (e.g. RandomForest) provides 
insight into the impact of each feature on reducing the model’s mean squared error (MSE).
The higher the importance value, the more frequently the model uses that feature for data 
splitting, and the greater its influence on predictions.

The only issue is that feature_importances_ returns only an array of numerical values without feature names.
To obtain the corresponding feature names, individual transformers from the ColumnTransformer pipeline can be used.

```
 Feature  Importance
       median_income    0.315142
              INLAND    0.153278
 population_per_hhold    0.108983
   bedrooms_per_hhold    0.083173
            longitude    0.076399
             latitude    0.067204
       rooms_per_hhold    0.062923
    housing_median_age    0.042631
           population    0.017808
          total_rooms    0.017451
       total_bedrooms    0.016637
           households    0.016605
          <1H OCEAN    0.013506
           NEAR OCEAN    0.004725
             NEAR BAY    0.003464
              ISLAND    0.000073
```

As can be seen from the table above, the most important feature is median_income —
which was already identified during the initial exploratory data analysis as having 
the highest correlation with housing prices.

Interestingly, the model also indicates that geographical features (ocean_proximity categories)
have a strong overall impact:
their combined importance reaches approximately 0.17, confirming that location remains a key 
factor affecting housing values.

## Model assessing
The final step of building the model is to test it on a separate dataset and evaluate its accuracy.
To do so, the RMSE metric can be used, which in this case equals:

```
47947.60
```

Since we cannot be sure how well this RMSE represents the true model error,
it makes sense to compute a 95% confidence interval as well:

```
[45941.9980476  49872.61787755]
```

As we can see, the model is not very accurate, as it shows a rather high prediction error
compared to the range of housing prices.

