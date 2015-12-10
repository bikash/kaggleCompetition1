import pandas as pd
import numpy as np

__author__ = 'Bikash'


def add_prefix(dataset, prefix):
    col_names = np.array(dataset.columns.values).astype('str')
    for i_in in range(col_names.shape[0]):
        col_names[i_in] = prefix + col_names[i_in]
    dataset.columns = col_names
    return dataset


def remove_sparse(dataset, n_valid_samples):
    col_names = list(dataset.columns.values)
    dead_cols = []
    for i_in in range(len(col_names)):
        if np.abs(np.sum(np.array(dataset[col_names[i_in]]))) < n_valid_samples:
            dead_cols.append(col_names[i_in])

    dataset = dataset.drop(dead_cols, axis=1)
    return dataset


def parse_rule(string, digit):
    return string[digit]
vec_parse_rule = np.vectorize(parse_rule)


def dummy_sep(df, col, bought, returned):
    dum = pd.get_dummies(df[col])
    tmp_index = dum.index
    tmp_columns = list(dum.columns.values)

    # separate b/r department variables
    # bought Department
    tmp_table = np.array(dum) * bought
    count_bought = pd.DataFrame(tmp_table)
    count_bought.columns = tmp_columns
    count_bought.index = tmp_index
    count_bought = count_bought.groupby(by=count_bought.index, sort=False).sum()
    count_bought = add_prefix(count_bought, col[:3] + '_B_')

    # returned Department
    tmp_table = np.array(dum) * returned
    count_returned = pd.DataFrame(tmp_table)
    count_returned.columns = tmp_columns
    count_returned.index = tmp_index
    count_returned = count_returned.groupby(by=count_returned.index, sort=False).sum()
    count_returned = add_prefix(count_returned, col[:3] + '_R_')
    return count_bought, count_returned


def dummy_sep_sparse(df, col, sparsity, bought, returned):
    col_density = df[col].value_counts()
    n_features = np.sum(col_density > sparsity)
    # print n_features
    col_density = col_density.iloc[:n_features]
    dummy_cols = list(col_density.index)

    # remove sparse products
    tmp_series = np.zeros((df.shape[0], 1))
    for i in range(df.shape[0]):
        col_number = df.iloc[i][col]
        if col_number in dummy_cols:
            tmp_series[i] = col_number
    df[col] = tmp_series
    print df[col].value_counts()

    # bought
    data_count = pd.get_dummies(df[col])
    tmp_index = data_count.index
    tmp_columns = list(data_count.columns.values)
    tmp_table = np.array(data_count) * np.array(bought)
    count_bought = pd.DataFrame(tmp_table)
    count_bought.columns = tmp_columns
    count_bought.index = tmp_index
    count_bought = count_bought.groupby(by=count_bought.index, sort=False).sum()
    count_bought = add_prefix(count_bought, col[:3] + '_B_')

    # returned
    data_count = pd.get_dummies(df[col])
    tmp_index = data_count.index
    tmp_columns = list(data_count.columns.values)
    tmp_table = np.array(data_count) * np.array(returned)
    count_returned = pd.DataFrame(tmp_table)
    count_returned.columns = tmp_columns
    count_returned.index = tmp_index
    count_returned = count_returned.groupby(by=count_returned.index, sort=False).sum()
    count_returned = add_prefix(count_returned, col[:3] + '_R_')
    return count_bought, count_returned


def max_digit(df, col):
    arr = np.array(df[col])
    max_len = 1
    for i in range(arr.shape[0]):
        if len(arr[i]) > max_len:
            max_len = len(arr[i])
    return max_len


def dummy_digits(df, col, n_dig):
    array = np.array(df[col])
    array = fill_zeros(array, n_dig)
    digit_dummies = []
    for i_in in range(n_dig):
        digit_list = list(array)
        digit_list = map(lambda x: x[i_in], digit_list)
        digit_list = pd.DataFrame(digit_list)
        digit_list.index = df.index
        digit_list.columns = [col + str(i_in)]
        digit_dummies.append(pd.get_dummies(digit_list))
    digit_dummies = pd.concat(digit_dummies, axis=1)
    digit_dummies = digit_dummies.groupby(by=digit_dummies.index, sort=False).sum()
    return digit_dummies


def fill_zeros(arr, n_digits):
    for i_in in range(arr.shape[0]):
        if len(arr[i_in]) < n_digits:
            arr[i_in] = arr[i_in] + str(10 ** (n_digits - len(arr[i_in])))[1:]
    return arr

"""
preprocessing data
"""
# preprocess test data
print 'read train data'
trainset = pd.DataFrame.from_csv('data/train.csv', index_col=1)
trainset = trainset.fillna('9999')
trainset[['Upc', 'FinelineNumber']] = trainset[['Upc', 'FinelineNumber']].astype(long)
trainset[['Upc', 'FinelineNumber']] = trainset[['Upc', 'FinelineNumber']].astype(str)
n = trainset.shape[0]
print 'train set record' 
print  n
fln_dig = max_digit(trainset, 'FinelineNumber')
upc_dig = max_digit(trainset, 'Upc')

train_result = trainset['TripType']
train_result = train_result.groupby(by=train_result.index, sort=False).mean()
print train_result.value_counts()
train_result.to_csv("data/train_result.csv")

n_trips = train_result.shape[0]

sparsity = 150

train_data_not_count = pd.get_dummies(trainset['Weekday'])
train_data_not_count = train_data_not_count.groupby(by=train_data_not_count.index, sort=False).mean()
train_data_not_count = train_data_not_count.astype('int')

train_data_fln_digits = dummy_digits(trainset, 'FinelineNumber', fln_dig)
train_data_fln_digits = train_data_fln_digits.astype('int')
train_data_upc_digits = dummy_digits(trainset, 'Upc', upc_dig)
train_data_upc_digits = train_data_upc_digits.astype('int')

# separate between returned and bought goods
train_total_items = np.array(trainset['ScanCount']).reshape((n, 1))
train_bought_items = np.clip(train_total_items, a_min=0, a_max=99999)
train_returned_items = np.clip(train_total_items, a_min=-99999, a_max=0)

train_dep_num_b = np.ones((train_result.shape[0], 1))
train_dep_num_b = pd.DataFrame(train_dep_num_b)
train_dep_num_b.index = train_result.index
train_dep_num_r = np.ones((train_result.shape[0], 1))
train_dep_num_r = pd.DataFrame(train_dep_num_r)
train_dep_num_r.index = train_result.index

train_fln_num_b = np.ones((train_result.shape[0], 1))
train_fln_num_b = pd.DataFrame(train_fln_num_b)
train_fln_num_b.index = train_result.index
train_fln_num_r = np.ones((train_result.shape[0], 1))
train_fln_num_r = pd.DataFrame(train_fln_num_r)
train_fln_num_r.index = train_result.index

train_upc_num_b = np.ones((train_result.shape[0], 1))
train_upc_num_b = pd.DataFrame(train_upc_num_b)
train_upc_num_b.index = train_result.index
train_upc_num_r = np.ones((train_result.shape[0], 1))
train_upc_num_r = pd.DataFrame(train_upc_num_r)
train_upc_num_r.index = train_result.index

train_dep_num_b.columns = ['dep_num_B']
train_dep_num_r.columns = ['dep_num_R']

train_fln_num_b.columns = ['fln_num_B']
train_fln_num_r.columns = ['fln_num_R']

train_upc_num_b.columns = ['upc_num_B']
train_upc_num_r.columns = ['upc_num_R']
indexes = list(train_result.index.values)

print 'Department counter'
for i in range(len(indexes)):
    single_vis = trainset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        train_dep_num_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            train_dep_num_b.loc[indexes[i]] = 1
        else:
            train_dep_num_b.loc[indexes[i]] = len(list(single_vis_bought['DepartmentDescription'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        train_dep_num_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            train_dep_num_r.loc[indexes[i]] = 1
        else:
            train_dep_num_r.loc[indexes[i]] = len(list(single_vis_returned['DepartmentDescription'].value_counts()))

print 'fln counter'
for i in range(len(indexes)):
    single_vis = trainset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        train_fln_num_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            train_fln_num_b.loc[indexes[i]] = 1
        else:
            train_fln_num_b.loc[indexes[i]] = len(list(single_vis_bought['FinelineNumber'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        train_fln_num_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            train_fln_num_r.loc[indexes[i]] = 1
        else:
            train_fln_num_r.loc[indexes[i]] = len(list(single_vis_returned['FinelineNumber'].value_counts()))

print 'upc counter'
for i in range(len(indexes)):
    single_vis = trainset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        train_upc_num_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            train_upc_num_b.loc[indexes[i]] = 1
        else:
            train_upc_num_b.loc[indexes[i]] = len(list(single_vis_bought['Upc'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        train_upc_num_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            train_upc_num_r.loc[indexes[i]] = 1
        else:
            train_upc_num_r.loc[indexes[i]] = len(list(single_vis_returned['Upc'].value_counts()))

print 'dummy train DepartmentDescription'
train_count_dep_bought, train_count_dep_returned = dummy_sep(trainset, 'DepartmentDescription',
                                                             train_bought_items, train_returned_items)

train_count_dep_bought = train_count_dep_bought.astype('int')
train_count_dep_returned = train_count_dep_returned.astype('int')

# find most bought FinelineNumber
print 'dummy train FinelineNumber'
train_count_fln_bought, train_count_fln_returned = dummy_sep_sparse(trainset, 'FinelineNumber', sparsity,
                                                                              train_bought_items, train_returned_items)
train_count_fln_bought = train_count_fln_bought.astype('int')
train_count_fln_returned = train_count_fln_returned.astype('int')

print 'dummy train Upc'
train_count_upc_bought, train_count_upc_returned = dummy_sep_sparse(trainset, 'Upc', sparsity,
                                                                              train_bought_items, train_returned_items)
train_count_upc_bought = train_count_upc_bought.astype('int')
train_count_upc_returned = train_count_upc_returned.astype('int')

train_bought_items = pd.DataFrame(train_bought_items)
train_bought_items.index = trainset.index
train_bought_items = train_bought_items.groupby(by=train_bought_items.index, sort=False).sum()
train_bought_items.columns = ['Bought']
train_bought_items = train_bought_items.astype('int')

train_returned_items = pd.DataFrame(train_returned_items)
train_returned_items.index = trainset.index
train_returned_items = train_returned_items.groupby(by=train_returned_items.index, sort=False).sum()
train_returned_items.columns = ['Returned']
train_returned_items = train_returned_items.astype('int')

train = pd.concat([train_data_not_count, train_count_dep_bought, train_count_dep_returned,
                   train_count_fln_bought, train_count_fln_returned,
                   train_count_upc_bought, train_count_upc_returned,
                   train_data_fln_digits, train_data_upc_digits,
                   train_dep_num_b, train_dep_num_r,
                   train_fln_num_b, train_fln_num_r,
                   train_upc_num_b, train_upc_num_r,
                   train_bought_items, train_returned_items], axis=1)
train = remove_sparse(train, sparsity * 0.25)

# preprocess test data
print 'read test data'
testset = pd.DataFrame.from_csv('data/test.csv', index_col=0)
testset = testset.fillna('9999')
testset[['Upc', 'FinelineNumber']] = testset[['Upc', 'FinelineNumber']].astype(long)
testset[['Upc', 'FinelineNumber']] = testset[['Upc', 'FinelineNumber']].astype(str)

test_data_not_count = pd.get_dummies(testset['Weekday'])
test_data_not_count = test_data_not_count.groupby(by=test_data_not_count.index, sort=False).mean()
test_data_not_count = test_data_not_count.astype('int')

n_test = testset.shape[0]
n_trips_test = test_data_not_count.shape[0]

test_data_fln_digits = dummy_digits(testset, 'FinelineNumber', fln_dig)
test_data_fln_digits = test_data_fln_digits.astype('int')
test_data_upc_digits = dummy_digits(testset, 'Upc', upc_dig)
test_data_upc_digits = test_data_upc_digits.astype('int')

test_dep_num_b = np.ones((test_data_not_count.shape[0], 1))
test_dep_num_b = pd.DataFrame(test_dep_num_b)
test_dep_num_b.index = test_data_not_count.index
test_dep_num_r = np.ones((test_data_not_count.shape[0], 1))
test_dep_num_r = pd.DataFrame(test_dep_num_r)
test_dep_num_r.index = test_data_not_count.index

test_fln_num_b = np.ones((test_data_not_count.shape[0], 1))
test_fln_num_b = pd.DataFrame(test_fln_num_b)
test_fln_num_b.index = test_data_not_count.index
test_fln_num_r = np.ones((test_data_not_count.shape[0], 1))
test_fln_num_r = pd.DataFrame(test_fln_num_r)
test_fln_num_r.index = test_data_not_count.index

test_upc_num_b = np.ones((test_data_not_count.shape[0], 1))
test_upc_num_b = pd.DataFrame(test_upc_num_b)
test_upc_num_b.index = test_data_not_count.index
test_upc_num_r = np.ones((test_data_not_count.shape[0], 1))
test_upc_num_r = pd.DataFrame(test_upc_num_r)
test_upc_num_r.index = test_data_not_count.index

test_dep_num_b.columns = ['dep_num_B']
test_dep_num_r.columns = ['dep_num_R']

test_fln_num_b.columns = ['fln_num_B']
test_fln_num_r.columns = ['fln_num_R']

test_upc_num_b.columns = ['upc_num_B']
test_upc_num_r.columns = ['upc_num_R']
indexes = list(test_data_not_count.index.values)

print 'Department counter'
for i in range(len(indexes)):
    single_vis = testset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        test_dep_num_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            test_dep_num_b.loc[indexes[i]] = 1
        else:
            test_dep_num_b.loc[indexes[i]] = len(list(single_vis_bought['DepartmentDescription'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        test_dep_num_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            test_dep_num_r.loc[indexes[i]] = 1
        else:
            test_dep_num_r.loc[indexes[i]] = len(list(single_vis_returned['DepartmentDescription'].value_counts()))

print 'fln counter'
for i in range(len(indexes)):
    single_vis = testset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        test_fln_num_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            test_fln_num_b.loc[indexes[i]] = 1
        else:
            test_fln_num_b.loc[indexes[i]] = len(list(single_vis_bought['FinelineNumber'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        test_fln_num_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            test_fln_num_r.loc[indexes[i]] = 1
        else:
            test_fln_num_r.loc[indexes[i]] = len(list(single_vis_returned['FinelineNumber'].value_counts()))

print 'upc counter'
for i in range(len(indexes)):
    single_vis = testset.loc[indexes[i]]
    bought = np.array(single_vis['ScanCount'] > 0)
    returned = np.array(single_vis['ScanCount'] < 0)
    single_vis_bought = single_vis.iloc[bought]
    single_vis_returned = single_vis.iloc[returned]

    if single_vis_bought.shape[0] == 0:
        test_upc_num_b.loc[indexes[i]] = 0
    else:
        if len(single_vis_bought.shape) == 1:
            test_upc_num_b.loc[indexes[i]] = 1
        else:
            test_upc_num_b.loc[indexes[i]] = len(list(single_vis_bought['Upc'].value_counts()))

    if single_vis_returned.shape[0] == 0:
        test_upc_num_r.loc[indexes[i]] = 0
    else:
        if len(single_vis_returned.shape) == 1:
            test_upc_num_r.loc[indexes[i]] = 1
        else:
            test_upc_num_r.loc[indexes[i]] = len(list(single_vis_returned['Upc'].value_counts()))

test_total_items = np.array(testset['ScanCount']).reshape((n_test, 1))
test_bought_items = np.clip(test_total_items, a_min=0, a_max=99999)
test_returned_items = np.clip(test_total_items, a_min=-99999, a_max=0)

print 'dummy test DepartmentDescription'
test_count_dep_bought, test_count_dep_returned = dummy_sep(testset, 'DepartmentDescription',
                                                           test_bought_items, test_returned_items)
test_count_dep_bought = test_count_dep_bought.astype('int')
test_count_dep_returned = test_count_dep_returned.astype('int')

print 'dummy test FinelineNumber'
test_count_fln_bought, test_count_fln_returned = dummy_sep_sparse(testset, 'FinelineNumber', sparsity,
                                                                  test_bought_items, test_returned_items)
test_count_fln_bought = test_count_fln_bought.astype('int')
test_count_fln_returned = test_count_fln_returned.astype('int')

print 'dummy test Upc'
test_count_upc_bought, test_count_upc_returned = dummy_sep_sparse(testset, 'Upc', sparsity,
                                                                  test_bought_items, test_returned_items)
test_count_upc_bought = test_count_upc_bought.astype('int')
test_count_upc_returned = test_count_upc_returned.astype('int')

test_bought_items = pd.DataFrame(test_bought_items)
test_bought_items.index = testset.index
test_bought_items = test_bought_items.groupby(by=test_bought_items.index, sort=False).sum()
test_bought_items.columns = ['Bought']
test_bought_items = test_bought_items.astype('int')

test_returned_items = pd.DataFrame(test_returned_items)
test_returned_items.index = testset.index
test_returned_items = test_returned_items.groupby(by=test_returned_items.index, sort=False).sum()
test_returned_items.columns = ['Returned']
test_returned_items = test_returned_items.astype('int')

test = pd.concat([test_data_not_count, test_count_dep_bought, test_count_dep_returned,
                  test_count_fln_bought, test_count_fln_returned,
                  test_count_upc_bought, test_count_upc_returned,
                  test_data_fln_digits, test_data_upc_digits,
                  test_dep_num_b, test_dep_num_r,
                  test_fln_num_b, test_fln_num_r,
                  test_upc_num_b, test_upc_num_r,
                  test_bought_items, test_returned_items], axis=1)
test = remove_sparse(test, sparsity * 0.25)

# Find common coloumns
col_train = list(train.columns.values)
print col_train
col_test = list(test.columns.values)
print col_test
col_common = []
# add only common columns for train and test
for col in col_train:
    if col in col_test:
        col_common.append(col)
train = train[col_common].astype('int')
test = test[col_common].astype('int')
print col_common

print 'write to data'
train.to_csv("data/train_feat.csv")
test.to_csv("data/test_feat.csv")
