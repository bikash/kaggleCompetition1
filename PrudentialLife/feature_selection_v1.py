import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Bikash'


def add_prefix(dataset, prefix):
    col_names = list(np.array(dataset.columns.values).astype('str'))
    for i_in in range(len(col_names)):
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


def dummy_2d(dataset, column):
    print '2d dummying %s' % column
    index = dataset.index
    dataset = list(dataset[column].astype('str'))
    dataset_1 = map(lambda x: ord('E') - ord(x[0]), dataset)
    dataset_2 = map(lambda x: x[1], dataset)
    dataset_1 = np.array(dataset_1).reshape((len(dataset_1), 1))
    dataset_2 = np.array(dataset_2).reshape((len(dataset_2), 1))
    dataset = pd.DataFrame(np.hstack((dataset_1, dataset_2)))
    dataset.index = index
    dataset = add_prefix(dataset, column)
    # print dataset
    return dataset


def dummy_num(dataset, column):
    print 'num dummying %s' % column
    index = dataset.index
    dataset = list(dataset[column].astype('str'))
    # Find max digits
    n_dig = 0
    for i in range(len(dataset)):
        if len(dataset[i]) > n_dig:
            n_dig = len(dataset[i])
    # Fill digits
    for i in range(len(dataset)):
        if len(dataset[i]) < n_dig:
            for j in range(n_dig - len(dataset[i])):
                dataset[i] = '0' + dataset[i]
    dataset_digits = []
    for i in range(n_dig):
        dataset_digits.append(map(lambda x: x[i], dataset))
        dataset_digits[i] = np.array(dataset_digits[i]).reshape((len(dataset_digits[i]), 1))
    dataset = pd.DataFrame(np.hstack(tuple(dataset_digits)))
    dataset.index = index
    dataset = add_prefix(dataset, column)
    # print dataset
    return dataset


"""
preprocessing data
"""
need_dummying = ['Product_Info_1', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6',
                 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1',
                 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
                 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4',
                 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1',
                 'Medical_History_3', 'Medical_History_4', 'Medical_History_5',
                 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9',
                 'Medical_History_10', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13',
                 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18',
                 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22',
                 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27',
                 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31',
                 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36',
                 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40',
                 'Medical_History_41']

# preprocess test data
print 'read train data'
trainset = pd.DataFrame.from_csv('data/train.csv', index_col=0)
train_index = trainset.index
train_result = trainset['Response']

print train_result.value_counts()
train_result.to_csv("data/train_result.csv")

trainset = trainset.drop('Response', axis=1)
trainset = trainset.fillna('-1')

# preprocess test data
print 'read test data'
testset = pd.DataFrame.from_csv('data/test.csv', index_col=0)
testset = testset.fillna('-1')
test_index = testset.index

data = pd.concat([trainset, testset])

# special dummies
p_info2_dummy = dummy_2d(data, 'Product_Info_2')
data = data.drop('Product_Info_2', axis=1)
m_history2_dummy = dummy_num(data, 'Medical_History_2')
data = data.drop('Medical_History_2', axis=1)

data = pd.concat([data, p_info2_dummy, m_history2_dummy], axis=1)

# location_ranking_1 = list(target1.location.value_counts().index)
# location_ranking_1_n = len(location_ranking_1)
# for loc in unique_locations:
#     if not(loc in location_ranking_1):
#         location_ranking_1.append(loc)
# location_ranking_1 = pd.Series(np.arange(len(location_ranking_1)), index=location_ranking_1)
# for i in range(location_ranking_1_n, unique_locations_n):
#     location_ranking_1.iat[i] = 2000
# print location_ranking_1
#
# full_data['location_sorted1'] = np.zeros((full_data.shape[0],))
# for i in range(len(full_data)):
#     cur_loc = full_data['location'].iloc[i]
#     full_data['location_sorted1'].iat[i] = location_ranking_1.loc[cur_loc]
# print full_data['location_sorted1']
train = data.loc[train_index]
test = data.loc[test_index]

# Plotting
for var in need_dummying:
    print 'var %s' % var
    print trainset[var].value_counts()
    print testset[var].value_counts()

print 'write to data'
train.to_csv("data/train_v4.csv")
test.to_csv("data/test_v4.csv")