"""
Preprocessing pipeline for housing data.

Steps:
- load the housing dataset.
- separate numerical and categorical attributes.
- build preprocessing pipelines:
   - numerical pipeline: missing values, combining attributes, scaling features
   - categorical pipeline: numerical + one-hot encoding
- transform the dataset

Output:
- `housing_num_tr`: transformed numerical data (NumPy array)
- `housing_prepared`: fully preprocessed dataset (sparse matrix)
"""

from scripts.chapter_1.data_loading import fetch_file_data
from transformer import CombinedAttributesAdder
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

housing_filename = "housing.csv"
housing_chapter = "chapter_1"
housing = fetch_file_data(housing_filename, housing_chapter)


housing_num = housing.select_dtypes(include=[np.number])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
print(type(housing_num_tr))

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)
print(type(housing_prepared))


