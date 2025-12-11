from pandas.plotting import scatter_matrix
import numpy as np


def get_data_info(dataset):
    """
    Prints short information for the given dataset

    Parameters:
    - dataset: DataFrame, dataset to investigate
    """
    print(dataset.head())
    print("\n" + "="*80)
    print(dataset.info())
    print("\n" + "=" * 80)
    # checking not number column
    print(dataset["ocean_proximity"].value_counts())
    print("\n" + "=" * 80)
    # getting description for numerical columns
    print(dataset.describe())
    # dataset.hist(bins=50, figsize=(20,15))
    # plt.show()

def explore_data(dataset):
    housing_copy = dataset.copy()
    # housing_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    # housing_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    #                   s=housing_copy["population"]/100, label="population", figsize=(10,7),
    #                   c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing_copy[attributes], figsize=(12,8))
    housing_copy.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

    housing_copy["rooms_per_household"] = housing_copy["total_rooms"] / housing_copy["households"]
    housing_copy["bedrooms_per_household"] = housing_copy["total_bedrooms"] / housing_copy["total_rooms"]
    housing_copy["population_per_household"] = housing_copy["population"] / housing_copy["households"]
    # attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    # housing_extra_attributes = attr_adder.transform(housing_copy.values)
    # print(housing_extra_attributes)
    numeric_data = housing_copy.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))