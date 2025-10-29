"""
Main entry point for Chapter 1 pipeline (based on Geron book).

Steps:
1. Load the housing dataset.
2. Explore and visualize the data.
3. Perform stratified split into train/test sets.
4. Prepare data for ML algorithms (handle missing values, encode categories).
"""

import numpy as np
import pandas as pd

from scripts.chapter_1.data_exploration import explore_data
from scripts.chapter_1.data_loading import fetch_housing_data
from scripts.chapter_1.data_preparation import prepare_dataset_for_ml
from scripts.chapter_1.data_splitting import stratify_dataset

if __name__ == "__main__":
    """Runs the full data preparation workflow for Chapter 1."""

    housing_filename = "housing.csv"
    housing_chapter = "chapter_1"
    housing = fetch_housing_data(housing_filename, housing_chapter)
    print("Data loaded")

    # indexing
    # housing_with_id = housing.reset_index()
    # get_data_info(housing)

    # custom splitting >> see data_splitting
    # simple splitting w/o stratification
    # train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

    # Create income category for stratified sampling
    housing["income_cat"] = pd.cut(housing["median_income"],
                                       bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                       labels=[1, 2, 3, 4, 5])
    # housing["income_cat"].hist()

    # stratifying
    strat_train_set, strat_test_set = stratify_dataset(housing, housing["income_cat"])
    # proportion = strat_test_set["income_cat"].value_counts()/len(strat_test_set)

    # remove tmp column
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Data exploration
    explore_data(strat_train_set)

    #     preparing dataset for ML algorithms
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    prepare_dataset_for_ml(housing)





