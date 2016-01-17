import sys
import numpy as np
import pandas as pd
import itertools
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
from sklearn import cross_validation as cv
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn import preprocessing

import toolkit

from StringIO import StringIO
import pydot

def get_train_data(file_train, scale=False, size=None):
    data = np.genfromtxt(fname=file_train, delimiter=',')
    np.random.shuffle(data)

    x = data[:size,1:-1]
    if scale:
        x = preprocessing.scale(x)
    y = data[:size, -1]
    return x, y


def get_test_data(file_test, scale=False):
    data = np.genfromtxt(fname=file_test, delimiter=',') 
    
    x, ids = data[:,1:], data[:,0]
    if scale:
        x = preprocessing.scale(x)
    return x, ids 


def test(classifier, x):
    preds = classifier.predict(x)
    return preds


def print_r(scale=True, size=3000):
    x, y = get_train_data('train.dat')
    classifier = RandomForestClassifier(n_estimators=100, 
                    max_features = 'auto', #min_samples_split=10,
                    criterion='gini', class_weight='balanced', n_jobs=-1)
    #scorer = make_scorer(toolkit.kaggle_scorer, 
    score = cv.cross_val_score(classifier, x, y, cv=6, scoring='f1_weighted')

    classifier = RandomForestClassifier(n_estimators=100, 
                    max_features = 'auto', #min_samples_split=10,
                    criterion='gini', class_weight='balanced', n_jobs=-1)
    classifier.fit(x[:size],y[:size])
    preds = classifier.predict(x[size:2*size])
    pred_probas = classifier.predict_proba(x[size:2*size])
    lb_score = toolkit.kaggle_scorer(y[size:2*size], preds, pred_probas)

    cm = confusion_matrix(y[size:2*size], preds)
    
    print score
    print lb_score
    print cm



def train(x, y, depth=None):
    rfc = RandomForestClassifier(n_estimators=100, 
                    min_samples_split=10, max_features = 'auto', 
                    criterion='gini', class_weight='balanced', n_jobs=-1)
    rfc.fit(x,y)
    return rfc



def main():
    x, y = get_train_data('train.dat')
    data, ids = get_test_data('test.dat')
    rfc = train(x, y)
    #evaluation(x,y,test_x,test_y)

    preds = test(rfc, data)
    print 'id,predict_0,predict_1,predict_2'
    '''
    for tid, label in toolkit.transform(ids, preds):
        print '%s,%s' % (tid, ','.join([np.str(item) for item in label]))
    '''
    probas = rfc.predict_proba(data)
    for idx in xrange(probas.shape[0]):
        print '%s,%s' % (int(ids[idx]), ','.join([np.str(item) for item in probas[idx]]))

if __name__ == '__main__':
    #main()
    print_r()
    
    
