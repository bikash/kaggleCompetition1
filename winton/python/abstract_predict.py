__author__ = 'Gavin.Chan'

import random
import pandas as pd
import numpy as np
import time

from sklearn import preprocessing
from collections import Counter

LOG_MODE = 0
TEST = 0
SUBMISSION = 0
IS_STANDARDIZED = 0

class BasePredict:
    def __init__(self):
        random.seed(123)
        self.train_data_filename = ''
        self.test_data_filename = ''
        self.submission_filename = ''
        self.features_index = list(range(1, 26))
        self.features_filtered_index = [3,5,6,8,9,11,12,13,14,15,16,17,18,19,22,23,24,25]
        # self.features_filtered_index = [3,6,8,11,12,14,15,17,18,19,22,23,24,25]
        self.catagorical_filtered_index = [5,9,13,16]
        self.returns_prev_days_index = [26, 27]
        self.returns_intraday_index = list(range(28, 207))
        self.returns_predict_index = list(range(147, 209))
        self.returns_next_days_index = [207, 208]
        self.weight_intraday_index = 209
        self.weight_daily_index = 210
        self.train_batch_index = []
        self.train_unbatch_index = []
        self.num_of_days = 0

        # Data
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.test_prediction = pd.DataFrame()
        self.predictors = pd.Series()

    @staticmethod
    def run(f):
        start = time.localtime()
        if LOG_MODE > 0:
            print("===================================================")
            print("Start running function %s " % f.__name__)
            print("Start time = %s)" % time.strftime("%c", start))

        f()

        end = time.localtime()
        if LOG_MODE > 0:
            print("Finished running function %s" % f.__name__)
            print("Run time = %s" % (time.mktime(end) - time.mktime(start)))

    def get_data(self):
        self.train_data = pd.read_csv(self.train_data_filename)
        self.test_data = pd.read_csv(self.test_data_filename)

        # Prepare batch and unbatch index
        count = self.train_data.shape[0]
        self.train_batch_index = range(0, int(count*2/3))
        self.train_unbatch_index = range(int(count*2/3), count)

        # Prepare test prediction
        self.num_of_days = self.test_data.shape[0]
        self.test_prediction = pd.DataFrame(index=range(1, self.num_of_days + 1),
                                            columns=range(1, 63))
        self.test_prediction = self.test_prediction.fillna(0)

    def clean_data(self):
        train_data = self.train_data.copy(True)
        test_data = self.test_data.copy(True)

        # Clean features
        median = train_data[self.features_index].median(axis = 0)
        for col in train_data.columns[self.features_index]:
            train_data[col] = train_data[col].fillna(median[col])
            test_data[col] = test_data[col].fillna(median[col])

        # Clean target
        for col in train_data.columns[self.returns_intraday_index]:
            train_data[col] = train_data[col].fillna(0)

        # try to transform the values to labels
        encoders = {}
        for col in train_data.columns[1:self.returns_prev_days_index[0]]:
            train = train_data[col].values
            if any(abs(train - train.round(0)) > 0.0001):
                continue

            encoders[col] = preprocessing.LabelEncoder()
            train = train.astype(np.int64)
            test = test_data[col].astype(np.int64)
            try:
                train_data[col] = encoders[col].fit_transform(train)
            except:
                print("Warning occured in col %s" % col)
                continue

            if encoders[col].classes_.shape[0] < 50:
                try:
                    test_data[col] = encoders[col].transform(test)
                except:
                    print("Test data has different labels with the train data at col %s" % col)
                    continue

        # fill the na value with the most appeared values in the columns
        for col in train_data.columns[1:self.returns_prev_days_index[0]]:
            if train_data[col].dtypes == 'int32' or train_data[col].dtypes == 'int64':
                train_column_data = self.train_data[col]
                test_column_data = self.test_data[col]
                train_column_data_notna = train_column_data[False == train_column_data.isnull()].round(0)
                train_column_data_notna = encoders[col].transform(train_column_data_notna)
                most_common = Counter(train_column_data_notna).most_common(1)[0][0]
                train_data.loc[train_column_data.isnull(),col] = most_common
                test_data.loc[test_column_data.isnull(),col] = most_common

        if IS_STANDARDIZED:
            for col in train_data.columns[1:self.returns_prev_days_index[0]]:
                if train_data[col].dtypes != 'int32' and train_data[col].dtypes != 'int64':
                    train_data[col] = (train_data[col] - train_data[col].mean())/train_data[col].std()
                    train_data[col] = ((train_data[col] > 0) * 1 - (train_data[col] < 0) * 1) * train_data[col].abs()**2
                    test_data[col] = (test_data[col] - test_data[col].mean())/test_data[col].std()

        self.train_data = train_data
        self.test_data = test_data


    def prepare_predictors(self):
        # Predict unbatch prediction
        train_unbatch_predict = pd.DataFrame(index=self.train_unbatch_index,
                                             columns=range(1, 63))
        train_unbatch_predict = train_unbatch_predict.fillna(0)

        error = self.evaluate_error(self.train_data.iloc[self.train_unbatch_index,:],
                                    train_unbatch_predict)
        print("%s : Unbatched error = %.4f" % (self.__class__.__name__, error))

    def predict(self):
        pass

    def generate_prediction(self):
        # Output the submission to csv
        submission = self.test_prediction.transpose().unstack()
        submission.index = ['_'.join([str(i) for i in s]) for s in submission.index]
        submission.to_csv(self.submission_filename, header=['Predicted'], index_label='Id')

    def evaluate_error(self, actual, predict):
        weight = pd.DataFrame(index=range(0, actual.shape[0]),
                                    columns=range(1, 63))
        abs_error = abs(actual.iloc[:,self.returns_predict_index].values - predict.values)
        weight.iloc[:,0:2]   = np.tile(actual.iloc[:,self.weight_daily_index], (2, 1)).T
        weight.iloc[:,2:60]  = np.tile(actual.iloc[:,self.weight_intraday_index], (58, 1)).T
        weight.iloc[:,60:62] = np.tile(actual.iloc[:,self.weight_daily_index], (2, 1)).T
        abs_error = abs_error * weight.values
        count = abs_error.shape[0] * abs_error.shape[1]
        return abs_error.sum()/count

    def evaluate_error_partition(self, actual, predict):
        weight = pd.DataFrame(index=range(0, actual.shape[0]),
                              columns=range(1, 63))
        abs_error = abs(actual.iloc[:,self.returns_predict_index].values - predict.values)
        abs_error_zero = abs(actual.iloc[:,self.returns_predict_index].values)
        weight.iloc[:,0:2]   = np.tile(actual.iloc[:,self.weight_daily_index], (2, 1)).T
        weight.iloc[:,2:60]  = np.tile(actual.iloc[:,self.weight_intraday_index], (58, 1)).T
        weight.iloc[:,60:62] = np.tile(actual.iloc[:,self.weight_daily_index], (2, 1)).T
        abs_error = (abs_error * weight.values).sum(axis = 0)
        abs_error_zero = (abs_error_zero * weight.values).sum(axis = 0)
        return abs_error_zero - abs_error

    def run_all(self):
        self.run(self.get_data)
        self.run(self.clean_data)
        self.run(self.prepare_predictors)

        if SUBMISSION:
            self.run(self.predict)
            self.run(self.generate_prediction)


class PredictSubmit(BasePredict):
    def __init__(self):
        BasePredict.__init__(self)
        self.train_data_filename = '../data/train.csv'
        self.test_data_filename = '../data/test.csv'
        self.submission_filename = '../data/submission.csv'


class PredictTest(BasePredict):
    def __init__(self):
        BasePredict.__init__(self)
        self.train_data_filename = '../data/train_trim.csv'
        self.test_data_filename = '../data/test_trim.csv'
        self.submission_filename = '../data/submission_trim.csv'


class Predict(PredictSubmit, PredictTest):
    def __init__(self):
        if TEST == 1:
            PredictTest.__init__(self)
        else:
            PredictSubmit.__init__(self)
