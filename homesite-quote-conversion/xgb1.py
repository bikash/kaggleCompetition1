
import pandas as pd
import numpy as np
import xgboost as xgb
import operator
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib
matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt

#seed = 260681

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

print("## Loading Data")
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')


print("## Data Processing")
y = train.QuoteConversion_Flag.values
train = train.drop('QuoteNumber', axis=1)
#test = test.drop('QuoteNumber', axis=1)

# Lets play with some dates
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek


test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)

train = train.fillna(-1)
test = test.fillna(-1)

print("## Data Encoding")
for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

features = [s for s in train.columns.ravel().tolist() if s != 'QuoteConversion_Flag']
print("Features: ", features)
#for f in sorted(set(features)):
#    print f
#exit()

print("## Training")
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "nthread":-1,
          "max_depth": 10,
          "subsample": 0.8,
          "colsample_bytree": 0.8,
          "eval_metric": "auc",
          "silent": 1,
          "seed": 1301
          }
num_boost_round = 500

print("Train a XGBoost model")
X_train, X_valid = train_test_split(train, test_size=0.01)
y_train = X_train['QuoteConversion_Flag']
y_valid = X_valid['QuoteConversion_Flag']
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=100)


print("## Creating Feature Importance Map")
ceate_feature_map(features)
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png',bbox_inches='tight',pad_inches=1)
df.to_csv("feature_importance.csv")

print("## Predicting test data")
preds = gbm.predict(xgb.DMatrix(test[features]),ntree_limit=50)
test["QuoteConversion_Flag"] = preds
test[['QuoteNumber',"QuoteConversion_Flag"]].to_csv('output/test_predictions.csv', index=False)

