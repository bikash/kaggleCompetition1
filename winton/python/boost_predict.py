__author__ = 'Gavin.Chan'

import math
from linear_predict import SklearnPredict
from sklearn import ensemble, cross_validation
from scipy import stats


class GradientBoostPredict(SklearnPredict):
    def __init__(self, params):
        SklearnPredict.__init__(self)
        self.params = params

    def fit(self, X, y):
        predictor = []

        kf = cross_validation.KFold(y.shape[0], n_folds=3, random_state=1)
        for itr, icv in kf:
            clf = ensemble.GradientBoostingRegressor(**(self.params))
            clf.fit(X.iloc[itr], y.iloc[itr].values)
            predictor.append(clf)

        return predictor

class FilterGradientBoostPredict(GradientBoostPredict):
    def __init__(self, params, range):
        GradientBoostPredict.__init__(self, params)
        self.range = range

    def batch_filter(self, X, Y):
        n, min_max, mean, var, skew, kurt = stats.describe(Y)
        sd = math.sqrt(var)
        y_index = Y[(Y > mean - self.range * sd).values & (Y < mean + self.range * sd).values].index.tolist()
        X = X.iloc[y_index, :]
        Y = Y[y_index]
        return X, Y
