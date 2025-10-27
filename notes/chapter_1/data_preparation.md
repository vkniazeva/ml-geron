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

