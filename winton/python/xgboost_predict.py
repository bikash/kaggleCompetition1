__author__ = 'Gavin.Chan'

from linear_predict import SklearnPredict
from sklearn import cross_validation
import xgboost as xgb
import pandas as pd
import math
from scipy import stats

class XGBoostPredict(SklearnPredict):
    def __init__(self, param):
        SklearnPredict.__init__(self)
        self.param = param

    def huber_obj(preds, dtrain):
        labels = dtrain.get_label()
        gamma = 1
        delta = preds - labels
        first_grad = ((delta < gamma) & (delta > -gamma)) * delta
        second_grad = (delta >= gamma) * gamma + (delta <= -gamma) * -gamma
        grad = first_grad + second_grad
        hess = ((delta < gamma) & (delta > -gamma)) * 1.0
        return grad, hess

    def absolute_obj(self, preds, dtrain):
        sigma1 = 0.51
        sigma2 = 0.00001
        labels = dtrain.get_label()
        print 'labels', labels
        delta = preds - labels
        grad = 1.0 * (delta > 0) - 1.0 * (delta < 0)
        hess = ((delta > -sigma1) & (delta < -sigma2)) * -delta**-1 + \
               ((delta >= -sigma2) & (delta <= sigma2)) * 1/sigma2 + \
               ((delta > sigma2) & (delta < sigma1)) * delta**-1
        return grad, hess

    def fit(self, X, y):
        #param = {'max_depth':3, 'eta':1.0, 'objective':'reg:linear'}
        #xgboost_predict = XGBoostPredict(param)
        predictor = []
        kf = cross_validation.KFold(y.shape[0], n_folds = 3, random_state = 1)
        print 'self ', self
        for itr, icv in kf:
            dtrain = xgb.DMatrix(X.iloc[itr].as_matrix(), label=y.iloc[itr].as_matrix())
            bst = xgb.train(params=self.param, dtrain=dtrain, num_boost_round=5, obj=self.absolute_obj)

            #bst = xgb.train(params=self.param, dtrain=dtrain, num_boost_round=2)

            predictor.append(bst)

        return predictor

    def evaluate_unbatch_error(self, X):
        X_matrix = xgb.DMatrix(X.as_matrix())
        train_unbatch_predict = pd.DataFrame(index=range(1, X.shape[0] + 1),
                                             columns=range(1, 63))
        train_unbatch_predict = train_unbatch_predict.fillna(0)
        print 'self ->', self
        prediction_index = range(61,63)
        for i in range(0, len(prediction_index)):
            y_predict = pd.DataFrame()

            for index, xgb_model in enumerate(self.predictors[i]):
                y_predict[index] = xgb_model.predict(X_matrix)

            y_final = y_predict.mean(axis = 1)
            train_unbatch_predict[prediction_index[i]] = pd.Series(y_final).values
         print 'ubatch ->', self    
        return train_unbatch_predict

class FilterXGBoostPredict(XGBoostPredict):
    def __init__(self, param):
        XGBoostPredict.__init__(self, param)
        self.range = param["range"]

    def batch_filter(self, X, Y):
        n, min_max, mean, var, skew, kurt = stats.describe(Y)
        sd = math.sqrt(var)
        y_index = Y[(Y > mean - self.range * sd).values & (Y < mean + self.range * sd).values].index.tolist()
        X = X.iloc[y_index, :]
        Y = Y[y_index]
        return X, Y

