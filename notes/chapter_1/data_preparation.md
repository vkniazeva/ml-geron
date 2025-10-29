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


