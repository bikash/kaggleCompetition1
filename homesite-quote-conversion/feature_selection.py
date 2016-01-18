

import sys
import argparse
import numpy as np
import pandas as pd
import os
import gc

def main():

    

    print('Build features from training data and test data')

    # Read training data and test data
    print('Read training data and test data')
    df_train_data = pd.read_csv('input/train.csv')
    df_test_data = pd.read_csv('input/test.csv')

    id_col = df_train_data.columns[0]
    date_col = df_train_data.columns[1]
    target_col = df_train_data.columns[2]

    # Merge the training data and test data for feature extraction
    print('Merge the training data and test data for feature extraction')
    df_test_data.insert(2, target_col, np.zeros_like(df_test_data[id_col]))
    df_all_data = pd.concat([df_train_data, df_test_data])
    train_test_split = len(df_train_data[id_col])
    del df_train_data
    del df_test_data
    gc.collect()

    # Separate feature columns and non-feature columns
    print('Separate feature columns and non-feature columns')
    df_id = df_all_data[[id_col]]
    df_date = pd.to_datetime(df_all_data[date_col])
    df_target = df_all_data[[target_col]]
    df_features = df_all_data[df_all_data.columns[3:]]
    del df_all_data
    gc.collect()

    # Transform date features
    print('Transform date features')
    date_features = pd.concat([df_date.dt.dayofweek, df_date.dt.day/df_date.dt.daysinmonth, df_date.dt.month], axis=1)
    date_features.columns = ['dt_dayofweek', 'dt_daysinmonth', 'dt_month']

    # Group features by types
    col_by_types = df_features.columns.to_series().groupby(df_features.dtypes).groups
    cat_cols = col_by_types[np.dtype('O')]
    int_cols = col_by_types[np.dtype('int64')]
    float_cols = col_by_types[np.dtype('float64')]

    # Cast to smaller data types
    print('Cast to smaller data types')
    for c in int_cols: df_features[c] = df_features[c].astype(np.int16)
    for c in float_cols: df_features[c] = df_features[c].astype(np.float16)

    # Fill NA for numerical features
    df_features['PersonalField84'] = df_features['PersonalField84'].fillna(0).astype(np.int16)
    df_features['PropertyField29'] = df_features['PropertyField29'].fillna(-1).astype(np.int16)

    # Find categorical feature with null values
    null_cols = df_features.columns[df_features.isnull().sum(axis=0) > 0]
    cat_null_cols = filter(lambda c: c in null_cols, cat_cols)
    cat_not_null_cols = filter(lambda c: c not in null_cols, cat_cols)

    # Get dummies for categorical features
    print('Get dummies for categorical features')
    df_cat_na_dummies = pd.get_dummies(df_features[cat_null_cols], dummy_na=True)
    df_cat_no_na_dummies = pd.get_dummies(df_features[cat_not_null_cols], dummy_na=False)

    # All available features
    train_test_dfs = [date_features, 
                      df_cat_na_dummies, 
                      df_cat_no_na_dummies, 
                      df_features[int_cols + float_cols], 
                      df_target] 

    # Remove all commas in column names just to be safe
    for df in train_test_dfs: 
        df.columns = map(lambda c: c.replace(',', ''), df.columns)

    # Split feature data into training set and test set
    # df_train_test_feature_target = pd.concat(train_test_dfs, axis=1)
    # df_train_test_feature_target.iloc[:train_test_split,:].to_csv(args.train_feature, index=False)
    # df_train_test_feature_target.iloc[train_test_split:,:].to_csv(args.test_feature, index=False)

    # Silly implementation of the same logic, to avoid memory copying
    print('Write feature files')
    for i, df in enumerate(train_test_dfs):
        df.iloc[:train_test_split].to_csv('data/train_feature' + str(i), index=False, header=True)
        df.iloc[train_test_split:].to_csv('data/test_feature' + str(i), index=False, header=True)
        del df
        gc.collect()

    print('Merge feature files for training data')
    train_feature_files = ['data/train_feature' + str(i) for i, df in enumerate(train_test_dfs)]
    train_file_cmd = 'paste -d , ' + ' '.join(train_feature_files) + ' > ' + 'data/train_feature'
    os.system(train_file_cmd)
    train_clean_cmd = 'rm ' + ' '.join(train_feature_files)
    os.system(train_clean_cmd)

    print('Merge feature files for test data')
    test_feature_files = ['data/test_feature' + str(i) for i, df in enumerate(train_test_dfs)]
    test_file_cmd = 'paste -d , ' + ' '.join(test_feature_files[:-1]) + ' > ' + 'data/test_feature'
    os.system(test_file_cmd)
    test_clean_cmd = 'rm ' + ' '.join(test_feature_files)
    os.system(test_clean_cmd)

if __name__ == '__main__':
    main()