import sys
import argparse
import numpy as np
import pandas as pd

import sklearn
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from xgboost import XGBClassifier

def main():
    model_name = 'XGBoost'



    print(model_name)

    # Read training data and test data
    print('Read training data and test data')
    df_train_feature_target = pd.read_csv('data/train_feature.csv', dtype=np.float64)
    df_test_feature = pd.read_csv('data/test_feature.csv', dtype=np.float64)

    train_X = df_train_feature_target.values[:,:-1]
    train_y = df_train_feature_target.values[:,-1]
    test_X = df_test_feature.values

    # Model specification and parameter range
    model = XGBClassifier(max_depth=10, learning_rate=0.025, silent=True, 
        subsample=0.8, colsample_bytree=0.8, nthread=-1)
    parameters = [{'n_estimators': [200, 100, 50, 25, 10]}]

    # Cross validation search
    print('Cross validation search')
    clf = GridSearchCV(model, parameters, 
        cv=5, scoring='roc_auc', n_jobs=4, pre_dispatch=4, verbose=3)
    clf.fit(train_X, train_y)

    # Make predictions with the best model
    print('Make predictions with the best model')
    train_pred = clf.predict(train_X)
    train_pred_prob = clf.predict_proba(train_X)[:,1]
    test_pred = clf.predict(test_X)
    test_pred_prob = clf.predict_proba(test_X)[:,1]

    # Write out the prediction result
    print('Write out the prediction result')
    prob ='prob'
    pd.Series(test_pred_prob  if prob else test_pred, name='Prob' if prob else 'Pred') \
        .to_csv('output/xgb.csv', index=False, header=True)

    # Report the result
    print('Report the result')
    print('Best Score: ', clf.best_score_)
    print('Best Parameter: ', clf.best_params_)
    print('Parameter Scores: ', clf.grid_scores_)
    print('Model: ', clf)
    print('Accuracy: ', accuracy_score(train_y, train_pred))
    print('F1:       ', f1_score(train_y, train_pred))
    print('ROC AUC:  ', roc_auc_score(train_y, train_pred_prob))
    print('output/xgb.csv' + '~~' + str(clf.best_score_))

if __name__ == '__main__':
    main()