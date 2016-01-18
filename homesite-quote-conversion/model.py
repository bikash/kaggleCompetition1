import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss,auc,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.cross_validation import  train_test_split
import xgboost as xgb
#load dataset
print 'loading dataset...'
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
print 'finish loading dataset...'

train = train.drop(['QuoteNumber'],axis=1)
train['Year'] = train['Original_Quote_Date'].apply(lambda x:int(str(x)[:4]))
train['Month'] = train['Original_Quote_Date'].apply(lambda x:int(str(x)[5:7]))
train['week'] = train['Original_Quote_Date'].apply(lambda x:int(str(x)[8:10]))
#train['Weekday'] = [train['Date'][i].dayofweek for i in range(len(train['Date']))]

test['Year'] = test['Original_Quote_Date'].apply(lambda x:int(str(x)[:4]))
test['Month'] = test['Original_Quote_Date'].apply(lambda x:int(str(x)[5:7]))
test['week'] = test['Original_Quote_Date'].apply(lambda x:int(str(x)[8:10]))
#test['Weekday'] = [test['Date'][i].dayofweek for i in range(len(test['Date']))]

train.drop(['Original_Quote_Date'],axis=1,inplace=True)
test.drop(['Original_Quote_Date'],axis=1,inplace=True)

#fill na
train.fillna(-1,inplace=True)
test.fillna(-1,inplace=True)

print 'preprocessing dataset...' 
#preprocess label
for f in train.columns:
    if train[f].dtype=='object':
        lbl=LabelEncoder()
        lbl.fit(np.unique(list(train[f].values)+list(test[f].values)))
        train[f]=lbl.transform(list(train[f].values))
        test[f]=lbl.transform(list(test[f].values))

Y_train = train['QuoteConversion_Flag']
X_train = train.drop(['QuoteConversion_Flag'],axis=1)
X_test = test.drop(['QuoteNumber'],axis=1).copy()

#xgboost
print 'training model...'
params = {"objective":"binary:logistic"}
T_train_xgb = xgb.DMatrix(X_train,Y_train)
x_test_xgb = xgb.DMatrix(x_test)
gbm = xgb.train(params,T_train_xgb,20)
Y_pred = gbm.predict(X_test_xgb)
print 'finish training model...'
#cross validation

#create submission
print 'create submission...'
submission =pd.DateFrame()
submission['QuoteNumber']=test['QuoteNumber']
submission['QuoteConversion_Flag']=Y_pred
submission.to_csv('output/submission_xgb.csv',index=False)
