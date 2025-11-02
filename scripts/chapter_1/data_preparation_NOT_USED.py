import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

def prepare_dataset_for_ml(dataset):
    # processing missing data
    dataset_num = dataset.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(dataset_num)
    # print(dataset_num.median().values)
    # print(imputer.statistics_)
    X = imputer.transform(dataset_num)
    dataset_tr = pd.DataFrame(X, columns=dataset_num.columns, index=dataset_num.index)
    # print(dataset_tr)

    #    categorical data processing
    dataset_cat = dataset[["ocean_proximity"]]
    # ordinal_encoder = OrdinalEncoder()
    # dataset_cat_encoded = ordinal_encoder.fit_transform(dataset_cat)
    # print(dataset_cat_encoded[:10])
    # print(ordinal_encoder.categories_)
    cat_encoder = OneHotEncoder()
    dataset_cat_1hot = cat_encoder.fit_transform(dataset_cat)
    dataset_cat_array = dataset_cat_1hot.toarray()
    print(dataset_cat_array)
    print(cat_encoder.categories_)