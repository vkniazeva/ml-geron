# Data preparation for ML algorithms

Data preparation typically includes:

* handling missing values
* processing categorical and text attributes
* feature scaling

## Data cleaning 

Most ML algorithms do not accept missing values, so they must be handled in advance.
Possible approaches:

* remove records with missing values — df.dropna(subset=[...])
* remove the entire attribute — df.drop([...], axis=1)
* fill missing values with a constant, mean, median, etc. — df.fillna(value=...)

A more robust approach is to use SimpleImputer from Scikit-Learn.
It can automatically compute replacement values for selected attributes 
and apply them consistently.

## Categorical data processing

When dealing with non-numerical data types, it is important to check whether they can be 
grouped into meaningful categories, since most ML algorithms prefer to work with numerical data.

There are several common ways to transform categorical data into numerical form, depending on their type:

For ordinal features (those with a meaningful order, e.g. “bad”, “average”, “good”), 
OrdinalEncoder is a suitable choice.
It assigns each category an integer value and returns a NumPy array.
In this case, ML algorithms treat the categories as ordered values

For nominal features (those without an inherent order), OneHotEncoder works better.
It creates one binary column per category — a one-hot encoded vector — 
resulting in an array filled with 0s and 1s.
To save memory, the encoder outputs a SciPy sparse matrix that can be converted to a dense NumPy array using .toarray().

```
[[0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0.]
```
For datasets with a large number of unique categories, one-hot encoding can decrease a model's performance —
it can significantly increase the number of features and slow down model training.
In such cases, one approach is to replace categorical labels with numerical representations
(e.g., instead of "ocean_proximity", use the actual distance to the ocean in kilometers).

Alternatively, embeddings can be used — learned vector representations where the model automatically discovers
an efficient encoding that helps predict the target variable.

## Feature scaling
Most ML algorithms are sensitive to how numeric features are scaled.
Some models like Decision Trees or Random Forests don’t really care,
but linear models, neural networks, and anything that relies on distance (like KNN or SVM) usually do.

The idea is simple:
if one feature has values from 0 to 10 and another from 0 to 10,000,
the model may treat the second one as more important just because of the bigger numbers — even if it’s not.

For example:

total_rooms: 6 → 39,320

median_income: 0 → 15

Without scaling, the model might think “rooms” matter much more than “income”, just because the range is bigger.

> 💡 The target variable usually doesn’t need scaling.

There are 2 common approaches:
* Min-Max scaling (Normalization)
    * MinMaxScaler 
    * (x - min(x)) / (max(x) - min(x))
    * Scales everything to the [0, 1] range
    * Works fine if there are no big outliers — otherwise, those can mess up the range.
* Standardization 
    * StandardScaler
    * (x- mean(x))/std 
    * After scaling, values are centered around 0 with standard deviation ≈ 1
    * Doesn’t limit the range but handles outliers better.

## Transformation pipelines

To simplify preprocessing steps (handling missing values, scaling, encoding, etc.),
scikit-learn provides the Pipeline class — it defines a sequence of transformations applied in order.

Each step is a tuple: ('name', estimator)

* The name must be unique and cannot contain "__".
* All intermediate estimators must implement fit_transform() (they learn and then transform the data).
* The last estimator usually implements only fit() or transform(), and produces the final output.

The ColumnTransformer class allows applying different pipelines to different feature subsets —
for example, numerical vs categorical.

It takes a DataFrame and expects a list of tuples:
(name, transformer, columns)

The output format depends on the transformers:
* if any transformer outputs a sparse matrix, the result will be sparse;
* otherwise, it will be dense.

Sparse matrices are preferable when:
* there are many categorical features (especially one-hot encoded)
* for linear models

Dense matrices are fine when:
* the dataset is small
* for tree-based models.

The best approach: to leave a matrix as it is and then to convert to dense if the model doesn’t support sparse input.








