import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from zlib import crc32

# def test_set_check(identifier, test_ratio):
#     """
#     Computes hash for a given value and checks it against test set selection criteria
#
#     :param identifier: value that will be computed as a hash function
#     :param test_ratio: proportion of a test set (float)
#     :return: hash value compared to the maximal 32-byte number (2**32) * test_ratio (boolean)
#     """
#     return crc32(np.int64(identifier)) & 0xffffffff < test_ratio*2**32
#
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

def split_train_test(dataset, test_ratio):
    shuffled_indices = np.random.permutation(len(dataset))
    test_set_size = int(len(dataset) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return dataset.iloc[train_indices], dataset.iloc[test_indices]

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



