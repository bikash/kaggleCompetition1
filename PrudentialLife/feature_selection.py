import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'WiBeer'


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
    dataset_1 = map(lambda x: x[0], dataset)
    dataset_2 = map(lambda x: x[1], dataset)
    dataset_1 = np.array(dataset_1).reshape((len(dataset_1), 1))
    dataset_2 = np.array(dataset_2).reshape((len(dataset_2), 1))
    dataset = pd.get_dummies(pd.DataFrame(np.hstack((dataset_1, dataset_2))))
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
    dataset = pd.get_dummies(pd.DataFrame(np.hstack(tuple(dataset_digits))))
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

chk_linearity = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4',
                 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4',
                 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15',
                 'Medical_History_24', 'Medical_History_32']

# preprocess test data
print 'read train data'
trainset = pd.DataFrame.from_csv('data/train.csv', index_col=0)
train_result = trainset['Response']

print train_result.value_counts()
train_result.to_csv("data/train_result.csv")

trainset_cols = list(trainset.columns.values)
trainset = trainset.drop('Response', axis=1)
trainset = trainset.fillna('-1')

# preprocess test data
print 'read test data'
testset = pd.DataFrame.from_csv('data/test.csv', index_col=0)
testset = testset.fillna('-1')

for col_name in chk_linearity:
    col = trainset[col_name].astype('float').values
    col_sqrd = col ** 2
    trainset[col_name + 'sqrd'] = col_sqrd
    col = testset[col_name].astype('float').values
    col_sqrd = col ** 2
    testset[col_name + 'sqrd'] = col_sqrd
    # plt.plot(col, train_result, 'ro')
    # plt.show()

# special dummies
train_p_info2_dummy = dummy_2d(trainset, 'Product_Info_2')
trainset = trainset.drop('Product_Info_2', axis=1)
test_p_info2_dummy = dummy_2d(testset, 'Product_Info_2')
testset = testset.drop('Product_Info_2', axis=1)

train_m_history2_dummy = dummy_num(trainset, 'Medical_History_2')
trainset = trainset.drop('Medical_History_2', axis=1)
test_m_history2_dummy = dummy_num(testset, 'Medical_History_2')
testset = testset.drop('Medical_History_2', axis=1)

# trainset[need_dummying] = trainset[need_dummying].astype(str)
n = trainset.shape[0]
# testset[need_dummying] = testset[need_dummying].astype(str)
n_test = testset.shape[0]

# sparsity = 0.95
# sparsity = n * (1 - sparsity)
# sparsity = n_test * (1 - sparsity)

# Plotting

print 'dummy train / test variables'
dummies = []
dummies_test = []
for var in need_dummying:
    print 'dummyfing %s' % var

    print trainset[var].value_counts()
    dummy_col = pd.get_dummies(trainset[var])
    add_prefix(dummy_col, var + '_')
    dummy_col.index = trainset.index
    dummies.append(dummy_col)
    trainset = trainset.drop(var, axis=1)

    print testset[var].value_counts()
    dummy_col = pd.get_dummies(testset[var])
    add_prefix(dummy_col, var + '_')
    dummy_col.index = testset.index
    dummies_test.append(dummy_col)
    testset = testset.drop(var, axis=1)

train = pd.concat([trainset, train_p_info2_dummy, train_m_history2_dummy] + dummies, axis=1)
# train = remove_sparse(train, sparsity * 0.25)

test = pd.concat([testset, test_p_info2_dummy, test_m_history2_dummy] + dummies_test, axis=1)
# test = remove_sparse(test, sparsity * 0.25)

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
print col_common

train = train[col_common]
test = test[col_common]

print 'write to data'
train.to_csv("data/train_dummied_v3.csv")
test.to_csv("data/test_dummied_v3.csv")

