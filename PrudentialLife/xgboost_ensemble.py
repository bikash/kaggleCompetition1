from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import LinearRegression
import xgboost as xgboost
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import StratifiedKFold, KFold
import scipy.optimize as optimize

__author__ = 'Bikash'


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))

    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


def ranking(predictions, split_index):
    predictions = pd.Series(predictions)
    ranked_predictions = predictions.copy()
    ranked_predictions.iloc[predictions < split_index[0]] = 1
    for i in range(1, (len(split_index) - 1)):
        ranked_predictions.iloc[split_index[i-1] <= predictions < split_index[i]] = i+1
    ranked_predictions.iloc[predictions >= split_index[-1]] = len(split_index-1)
    return ranked_predictions


def ranking(predictions, split_index):
    # print predictions
    ranked_predictions = np.ones(predictions.shape)

    for i in range(1, len(split_index)):
        cond = (split_index[i-1] <= predictions) * 1 * (predictions < split_index[i])
        ranked_predictions[cond.astype('bool')] = i+1
    cond = (predictions >= split_index[-1])
    ranked_predictions[cond] = len(split_index) + 1
    # print cond
    # print ranked_predictions
    return ranked_predictions


train_result = pd.DataFrame.from_csv("data/train_result.csv")
print train_result['Response'].value_counts()

col = list(train_result.columns.values)
result_ind = list(train_result[col[0]].value_counts().index)
train_result = np.array(train_result).ravel()

# combining meta_estimators
train = glob.glob('meta_train*')
train = sorted(train)
print train
for i in range(len(train)):
    train[i] = pd.DataFrame.from_csv(train[i])
train = pd.concat(train, axis=1)
train = np.array(train)

test = glob.glob('meta_test*')
test = sorted(test)
print test
for i in range(len(test)):
    test[i] = pd.DataFrame.from_csv(test[i])
test = pd.concat(test, axis=1)
test = np.array(test)

# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

# 4th
# splitter = [2.46039684, 3.48430979, 4.30777339, 4.99072484, 5.59295844, 6.17412558, 6.79373477]
# nelder mead opt
splitter_old = np.array([2.46039684, 3.48430979, 4.30777339, 4.99072484, 5.59295844, 6.17412558, 6.79373477])
riskless_splitter = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])


def opt_cut_global(predictions, results):
    print 'start quadratic splitter optimization'
    x0_range = np.arange(0, 4.25, 0.25)
    x1_range = np.arange(0.6, 1.5, 0.15)
    x2_range = np.arange(-0.08, 0.01, 0.01)
    bestcase = np.array(ranking(predictions, riskless_splitter)).astype('int')
    bestscore = quadratic_weighted_kappa(results, bestcase)
    print ('The starting score is %f' % bestscore)

    # optimize classifier
    for x0 in x0_range:
        for x1 in x1_range:
            for x2 in x2_range:
                case = np.array(ranking(predictions, (x0 + x1 * riskless_splitter + x2 *
                                                      riskless_splitter**2))).astype('int')
                score = quadratic_weighted_kappa(results, case, 1, 8)
                if score > bestscore:
                    bestscore = score
                    best_splitter = x0 + x1 * riskless_splitter + x2 * riskless_splitter**2
                    print 'For splitter ', (x0 + x1 * riskless_splitter + x2 * riskless_splitter**2)
                    print 'Variables x0 = %f, x1 = %f, x2 = %f' % (x0, x1, x2)
                    print 'The score is %f' % bestscore
    return best_splitter


def opt_cut_local(x, *args):
    predictions, results = args
    case = np.array(ranking(predictions, x)).astype('int')
    score = -1 * quadratic_weighted_kappa(results, case, 1, 8)
    # print score
    return score

best_risk = 0
best_score = 0
best_splitter = 0

param_grid = [
              {'silent': [1], 'nthread': [3], 'eval_metric': ['rmse'], 'eta': [0.01],
               'objective': ['reg:linear'], 'max_depth': [7], 'num_round': [900], 'fit_const': [0.5],
               'subsample': [0.75], 'risk': [0.5, 0.7, 0.9, 1]}
             ]

print 'start CV'
for params in ParameterGrid(param_grid):
    print params

    # CV
    cv_n = 10
    kf = KFold(train.shape[0], n_folds=cv_n, shuffle=True)
    it_splitter = []
    metric = []
    train_test_predictions = np.ones((train.shape[0],))
    for train_index, test_index in kf:
        X_train, X_test = train[train_index, :], train[test_index, :]
        y_train, y_test = train_result[train_index].ravel(), train_result[test_index].ravel()
        # train machine learning
        xg_train = xgboost.DMatrix(X_train, label=y_train)
        xg_test = xgboost.DMatrix(X_test, label=y_test)

        watchlist = [(xg_train, 'train'), (xg_test, 'test')]

        num_round = params['num_round']
        xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

        # predict
        predicted_results = xgclassifier.predict(xg_test)
        predicted_self_results = xgclassifier.predict(xg_train)
        splitter = opt_cut_global(predicted_self_results, y_train)
        res = optimize.minimize(opt_cut_local, splitter, args=(predicted_self_results, y_train), method='Nelder-Mead',
                                # options={'disp': True}
                                )
        train_test_predictions[test_index] = predicted_results
        # train machine learning

        # print res.x
        cur_splitter = list(params['risk'] * res.x + (1 - params['risk']) * riskless_splitter)
        it_splitter.append(cur_splitter)
        # print cur_splitter
        classified_predicted_results = np.array(ranking(predicted_results, cur_splitter)).astype('int')
        # predict
        print quadratic_weighted_kappa(y_test, classified_predicted_results, 1, 8)
        metric.append(quadratic_weighted_kappa(y_test, classified_predicted_results, 1, 8))
    print 'The quadratic weighted kappa is: ', np.mean(metric)
    if np.mean(metric) > best_score:
        print 'new best risk'
        best_score = np.mean(metric)
        best_risk = params['risk']
        it_splitter = np.array(it_splitter)
        best_splitter = np.average(it_splitter, axis=0)

print best_risk, best_score
# pd.DataFrame(train_test_predictions).to_csv('ensemble_train_predictions_xgboost_v3.csv')
print 'Calculating final splitter'
splitter = opt_cut_global(train_test_predictions, train_result)
# train machine learning
res = optimize.minimize(opt_cut_local, splitter, args=(train_test_predictions, train_result), method='Nelder-Mead',
                        # options={'disp': True}
                        )
classified_predicted_results = np.array(ranking(train_test_predictions, res.x)).astype('int')
print pd.Series(classified_predicted_results).value_counts()
print quadratic_weighted_kappa(train_result, classified_predicted_results, 1, 8)

# train machine learning
xg_train = xgboost.DMatrix(train, label=train_result)
xg_test = xgboost.DMatrix(test)

watchlist = [(xg_train, 'train')]

num_round = params['num_round']

xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

# predict
predicted_results = xgclassifier.predict(xg_test)
pd.DataFrame(predicted_results).to_csv('output/ensemble_test_predictions_xgboost_v3.csv')

predicted_self_results = xgclassifier.predict(xg_train)
pd.DataFrame(predicted_results).to_csv('output/ensemble_train_predictions_xgboost_v3.csv')
splitter = opt_cut_global(predicted_self_results, train_result)
res = optimize.minimize(opt_cut_local, splitter, args=(predicted_self_results, train_result), method='Nelder-Mead',
                        options={'disp': True}
                        )
splitter = list(best_risk * res.x + (1 - best_risk) * riskless_splitter)
print splitter
print 'writing to file'
classed_results = np.array(ranking(predicted_results, splitter)).astype('int')
submission_file = pd.DataFrame.from_csv("output/sample_submission.csv")
submission_file['Response'] = classed_results

print submission_file['Response'].value_counts()

submission_file.to_csv("output/ensemble_xgboost.csv")




