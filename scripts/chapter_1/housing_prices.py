import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


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

if __name__ == "__main__":
    housing_filename = "housing.csv"
    housing_chapter = "chapter_1"
    try:
        housing = fetch_housing_data(housing_filename, housing_chapter)
        # housing_with_id = housing.reset_index()
        get_data_info(housing)
        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
        # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
        housing["income_cat"] = pd.cut(housing["median_income"],
                                       bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                       labels=[1, 2, 3, 4, 5])
        housing["income_cat"].hist()
        plt.show()
        print(test_set)
    except Exception as e:
        print(e)


