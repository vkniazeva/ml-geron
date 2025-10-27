import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zlib import crc32
from pandas.plotting import scatter_matrix
from scipy.stats import alpha
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer


def fetch_housing_data(filename, chapter):
    """
    Load housing data from the assets folder.

    Parameters:
    - filename: str, name of the CSV file
    - chapter: str, chapter folder inside assets

    Returns:
    - pandas.DataFrame with the housing data
    """

    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    csv_path = os.path.join(project_root, "assets", chapter, filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        data = pd.read_csv(csv_path)
        return data
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {csv_path} is empty.")
    except pd.errors.ParserError:
        raise ValueError(f"The file {csv_path} could not be parsed.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading {csv_path}: {e}")


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


def split_train_test(dataset, test_ratio):
    shuffled_indices = np.random.permutation(len(dataset))
    test_set_size = int(len(dataset) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return dataset.iloc[train_indices], dataset.iloc[test_indices]


# def test_set_check(identifier, test_ratio):
#     """
#     Computes hash for a given value and checks it against test set selection criteria
#
#     :param identifier: value that will be computed as a hash function
#     :param test_ratio: proportion of a test set (float)
#     :return: hash value compared to the maximal 32-byte number (2**32) * test_ratio (boolean)
#     """
#     return crc32(np.int64(identifier)) & 0xffffffff < test_ratio*2**32

# def split_train_test_by_id(data, test_ratio, id_column):
#     """
#     Splits a given dataset based on a provided proportion by a specific column
#
#     :param data: dataset (exp. DataFrame)
#     :param test_ratio:  proportion of a test set (float)
#     :param id_column: column to split by
#     :return:
#     """
#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_:test_set_check(id_, test_ratio))
#     return data.loc[~in_test_set], data.loc[in_test_set]


def stratify_dataset(dataset, feature):
    """
    Splits dataset by stratifying it by a given column

    :param dataset: dataset to split (DataFrame)
    :param feature: to stratify by (array)
    :return: 2 datasets - train and test
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(dataset, feature):
        strat_train_set = dataset.loc[train_index]
        strat_test_set = dataset.loc[test_index]
        return strat_train_set, strat_test_set
    return None


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
    numeric_data = housing_copy.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))


def prepare_dataset_for_ml(dataset):
     dataset_num = dataset.drop("ocean_proximity", axis=1)
     imputer = SimpleImputer(strategy="median")
     imputer.fit(dataset_num)
     print(dataset_num.median().values)
     print(imputer.statistics_)
     X = imputer.transform(dataset_num)
     dataset_tr = pd.DataFrame(X, columns=dataset_num.columns, index=dataset_num.index)
     print(dataset_tr)



if __name__ == "__main__":
    housing_filename = "housing.csv"
    housing_chapter = "chapter_1"
    try:
        housing = fetch_housing_data(housing_filename, housing_chapter)
        # housing_with_id = housing.reset_index()
        # get_data_info(housing)

        # simple splitting w/o stratification
        # train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
        # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

        housing["income_cat"] = pd.cut(housing["median_income"],
                                       bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                       labels=[1, 2, 3, 4, 5])
        # housing["income_cat"].hist()

        strat_train_set, strat_test_set = stratify_dataset(housing, housing["income_cat"])
        # proportion = strat_test_set["income_cat"].value_counts()/len(strat_test_set)
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
        # explore_data(strat_train_set)

    #     preparing dataset for ML algorithms
        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()
        prepare_dataset_for_ml(housing)


    except Exception as e:
        print(e)


