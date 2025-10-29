from sklearn.base import BaseEstimator, TransformerMixin
import numpy

room_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Converts combined features

    inherits:
    - BaseEstimator - allows using get_params(), set_params()
    - TransformerMixin - adds fit_transform()
    """
    def __init__(self, add_bedrooms_per_room = True):
        """
        no *ars, no **kargs
        :param add_bedrooms_per_room: boolean - default: True, controls if to compute add_bedrooms_per_room feature
        """
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        """
        default implementation to match Transformer implementation
        can be self.mean_ = X.mean(axis=0)
        :param X: - DataFrame
        :param y: None as default
        :return: self
        """
        return self
    def transform(self, X, y=None):
        """
        Transforms data based on implemented internal computations

        :param X: DataFrame
        :param y: None as default
        :return: new DataFrame
        """
        rooms_per_household = X[:, room_ix] / X[:,households_ix]
        population_per_household = X[:, population_ix] / X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:,room_ix]
            return numpy.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return numpy.c_[X, rooms_per_household, population_per_household]



