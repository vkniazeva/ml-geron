import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    dataset.hist(bins=50, figsize=(20,15))
    # plt.show()

def split_train_test(dataset, test_ratio):
    shuffled_indices = np.random.permutation(len(dataset))
    test_set_size = int(len(dataset) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return dataset.iloc[train_indices], dataset.iloc[test_indices]



if __name__ == "__main__":
    housing_filename = "housing.csv"
    housing_chapter = "chapter_1"
    try:
        housing = fetch_housing_data(housing_filename, housing_chapter)
        get_data_info(housing)
    except Exception as e:
        print(e)


