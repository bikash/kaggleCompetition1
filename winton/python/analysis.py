__author__ = 'Gavin.Chan'

from abstract_predict import Predict
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats


class Analysis(Predict):
    def __init__(self):
        Predict.__init__(self)

    def describe_out(self):
        stat_train = pd.DataFrame(columns=['Min', 'Max', 'Mean', 'Median', 'SD', 'Skew', 'Kurt'])
        for i in range(self.features_index[0], self.returns_next_days_index[1]+1):
            data = self.train_data.iloc[:,i]
            n, min_max, mean, var, skew, kurt = stats.describe(data)
            stat_train.loc[self.train_data.columns[i]] = [min_max[0], min_max[1], mean, data.median(), scipy.sqrt(var), skew, kurt]
        stat_train.to_csv('../data/stat_train.csv')

        stat_test = pd.DataFrame(columns=['Min', 'Max', 'Mean', 'Median', 'SD', 'Skew', 'Kurt'])
        for i in range(self.features_index[0], self.returns_predict_index[0]):
            data = self.test_data.iloc[:,i]
            n, min_max, mean, var, skew, kurt = stats.describe(data)
            stat_test.loc[self.test_data.columns[i]] = [min_max[0], min_max[1], mean, data.median(), scipy.sqrt(var), skew, kurt]
        stat_test.to_csv('../data/stat_test.csv')

    def describe(self, s):
        n, min_max, mean, var, skew, kurt = stats.describe(s)
        print("Minimun: %.6f" % min_max[0])
        print("Maximum: %.6f" % min_max[1])
        print("Mean: %.6f" % mean)
        print("Standard derivation: %.6f" % scipy.sqrt(var))
        print("Skew: %.6f" % skew)
        print("Kurt: %.6f" % kurt)

    def return_histogram(self, index, diff):
        plt.figure()
        self.train_data.iloc[:, index].hist(bins=20)
        plt.show()

    def run_analysis(self):
        self.get_data()
        self.clean_data()
        self.describe_out()
        # self.describe(self.train_data.iloc[:, self.returns_prev_days_index[0]])
        # self.return_histogram(self.returns_prev_days_index[0], 0.01)
