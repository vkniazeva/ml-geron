import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from scripts.chapter_1.features.transformer import CombinedAttributesAdder

def build_full_pipeline(data):
    num_features = list(data.select_dtypes(include=[np.number]).columns)
    cat_features = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", OneHotEncoder(), cat_features),
    ])
    return full_pipeline
