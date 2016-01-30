import pandas as pd
import numpy as np


event_type = pd.read_csv('data/event_type.csv')
resource_type = pd.read_csv('data/resource_type.csv')
severity_type = pd.read_csv('data/severity_type.csv')
log_feature = pd.read_csv('data/log_feature.csv')


def filter_on_total(x, distribution, total=1, value='Rare'):

    try:
        if distribution[x] < total:
            return value
    except (ValueError, KeyError):
        return np.nan
    else:
        return x


def resource_type_features(df):

    resource_type_dummies = pd.get_dummies(resource_type['resource_type'])
    resource_dummy = pd.concat([resource_type, resource_type_dummies],
                               axis=1)

    resource_grpd = resource_dummy \
        .groupby(resource_dummy.id).sum()

    df_resource = pd.merge(df, resource_grpd, left_on='id', right_index=True)
    return df_resource.drop(['resource_type 1', 'resource_type 2',
                             'resource_type 4',
                             'resource_type 6', 'resource_type 8',
                             'resource_type 10'], axis=1)


def resource_type_event_count(df):

    p = pd.merge(resource_type, event_type, on='id')
    t = p.loc[p['resource_type'] == 'resource_type 2', ['id', 'event_type']] \
        .groupby(by=['id'], as_index=False).count()
    df_two = pd.merge(df, t, on='id', how='left')

    f = p.loc[p['resource_type'] == 'resource_type 4', ['id', 'event_type']] \
        .groupby(by='id', as_index=False).count()
    df_four = pd.merge(df_two, f, on='id', how='left')

    e = p.loc[p['resource_type'] == 'resource_type 8', ['id', 'event_type']] \
        .groupby(by='id', as_index=False).count()
    df_eight = pd.merge(df_four, e, on='id', how='left')

    t = p.loc[p['resource_type'] == 'resource_type 10', ['id', 'event_type']] \
        .groupby(by='id', as_index=False).count()
    df_ten = pd.merge(df_eight, t, on='id', how='left')

    return df_ten


def resource_type_log(df):

    p = pd.merge(resource_type, log_feature, on='id')
    t = p.loc[(p['resource_type'] == 'resource_type 1'), ['id', 'log_feature']] \
        .groupby(by='id', as_index=False).count()
    df_one = pd.merge(df, t, on='id', how='left')

    tw = p.loc[(p['resource_type'] == 'resource_type 2'), ['id', 'log_feature']] \
        .groupby(by='id', as_index=False).count()
    df_two = pd.merge(df_one, tw, on='id', how='left')

    se = p.loc[(p['resource_type'] == 'resource_type 7'), ['id', 'log_feature']] \
        .groupby(by='id', as_index=False).count()
    df_se = pd.merge(df_two, se, on='id', how='left')

    ei = p.loc[(p['resource_type'] == 'resource_type 8'), ['id', 'log_feature']] \
        .groupby(by='id', as_index=False).count()
    df_ei = pd.merge(df_se, ei, on='id', how='left')

    n = p.loc[(p['resource_type'] == 'resource_type 9'), ['id', 'log_feature']] \
        .groupby(by='id', as_index=False).count()
    df_n = pd.merge(df_ei, n, on='id', how='left')

    return df_n


def resource_type_log_sum(df):

    p = pd.merge(resource_type, log_feature, on='id')
    t = p.loc[(p['resource_type'] == 'resource_type 3'), ['id', 'volume']] \
        .groupby(by='id', as_index=False).sum()
    df_one = pd.merge(df, t, on='id', how='left')

    f = p.loc[(p['resource_type'] == 'resource_type 4'), ['id', 'volume']] \
        .groupby(by='id', as_index=False).sum()
    df_f = pd.merge(df, f, on='id', how='left')

    return df_f


def resource_type_log_total(df):

    p = pd.merge(resource_type, log_feature, on='id')

    # resource type 2
    t = p.loc[(p['resource_type'] == 'resource_type 2'), ['id', 'log_feature']] \
        .groupby(by='id', as_index=False).count()

    t.loc[:, 'volume'] = p.loc[(p['resource_type'] == 'resource_type 2'), ['id', 'volume']] \
        .groupby(by='id', as_index=False).sum()['volume']

    t.loc[:, 'total log volume two'] = t['volume'] / t['log_feature']

    df_one = pd.merge(df, t[['id', 'total log volume two']], on='id', how='left')


    # resource type ten
    te = p.loc[(p['resource_type'] == 'resource_type 10'), ['id', 'log_feature']] \
        .groupby(by='id', as_index=False).count()

    te.loc[:, 'volume'] = p.loc[(p['resource_type'] == 'resource_type 10'), ['id', 'volume']] \
        .groupby(by='id', as_index=False).sum()['volume']

    te.loc[:, 'total log volume ten'] = te['log_feature'] * te['volume']

    df_te = pd.merge(df_one, te[['id', 'total log volume ten']], on='id', how='left')

    return df_te


def resource_type_severity(df):

    p = pd.merge(resource_type, severity_type, on='id')
    p.loc[:, 'resource type severity'] = \
        ((p['resource_type'] == 'resource_type 6') &
        (p['severity_type'] == 'severity_type 1')).astype(float)

    t = p[['id', 'resource type severity']].groupby(by='id', as_index=False).median()

    return pd.merge(df, t, on='id', how='left')

    # events = [
    #     'event_type 38',
    #     'event_type 28',
    #     'event_type 23',
    #     'event_type 3',
    # ]

    # p = pd.merge(resource_type, event_type, on='id', how='left')
    # p.loc[:, 'resource 1 event'] = \
    #     (p['resource_type'] == 'resource_type 1') & (p['event_type'].isin(events)).astype(float)

    # t = p.groupby(by='id', as_index=False).mean()
    # return pd.merge(df, t, on='id', how='left')


# def resource_type_3_9_10(df):

#     r = resource_type[['id', 'resource_type']]

#     r.loc[:, 'resource type 3 9 10'] = \
#         (r['resource_type'].isin(['resource_type 3',
#                                   'resource_type 9',
#                                   'resource_type 10'])).astype(float)
#     ret = r.groupby(by='id', as_index=False).median()
#     return pd.merge(df, ret[['id', 'resource type 3 9 10']], on='id', how='left')

def severity_type_features(df):

    severity_type_dummies = pd.get_dummies(severity_type['severity_type'])
    severity_dummy = pd.concat([severity_type, severity_type_dummies],
                               axis=1)

    severity_grpd = severity_dummy \
        .groupby(severity_dummy.id).sum()

    df_severity = pd.merge(df, severity_grpd, left_on='id', right_index=True)
    return df_severity


def severity_high(df):

    p = severity_type[['id', 'severity_type']]

    p.loc[:, 'high log severity'] = \
        (p['severity_type'].isin(['severity_type 3',
                                   'severity_type 4',
                                   'severity_type 5'])).astype(float)
    ret = pd.merge(df, p[['id', 'high log severity']], on='id', how='left')
    return ret


def severity_event(df):

    p = pd.merge(severity_type, event_type, on='id')
    t = p.loc[(p['severity_type'] == 'severity_type 1'), ['id', 'event_type']] \
        .groupby(by='id', as_index=False).count()
    df_one = pd.merge(df, t, on='id', how='left')

    return df_one


def severity_log(df):

    p = pd.merge(severity_type, log_feature, on='id')

    tw = p.loc[(p['severity_type'] == 'severity_type 2'), ['id', 'log_feature']] \
        .groupby(by='id', as_index=False).count()
    df_tw = pd.merge(df, tw, on='id', how='left')

    return df_tw


def severity_log_sum(df):

    p = pd.merge(severity_type, log_feature, on='id')

    t = p.loc[(p['severity_type'] == 'severity_type 2'), ['id', 'log_feature']] \
        .groupby(by='id', as_index=False).count()
    t.loc[:, 'severity_volume'] = p.loc[(p['severity_type'] == 'severity_type 2'), ['id', 'volume']] \
        .groupby(by='id', as_index=False).sum()

    t.loc[:, 'severity_log_volume'] = t['severity_volume'] / t['log_feature']
    df_one = pd.merge(df, t[['id', 'severity_log_volume']], on='id', how='left')

    # tw = p.loc[(p['severity_type'] == 'severity_type 2'), ['id', 'volume']] \
    #     .groupby(by='id', as_index=False).sum()
    # df_tw = pd.merge(df, tw, on='id', how='left')

    return df_one


def log_features(df):

    log_table = pd.pivot_table(
        log_feature, values='volume',
        index='id', columns='log_feature',
        aggfunc=np.sum, fill_value=0
    )
    df_log = pd.merge(df, log_table, left_on='id', right_index=True)

    # print()
    # print(df_log.shape)
    # print()
    #df_log = pd.merge(df, log_feature, on='id')
    # print()
    # print(df_log.shape)
    # print()
    return df_log


def log_feature_volume(df):

    g = log_feature[['id', 'volume']].groupby(by='id', as_index=False).sum()
    ret_g = pd.merge(df, g, on='id', how='left')

    h = log_feature[['id', 'log_feature']].groupby(by='id', as_index=False).count()
    i = pd.merge(h, g, on='id', how='left')

    i.loc[:, 'feature_by_volume'] = i['log_feature'] * i['volume']
    ret = pd.merge(ret_g, i[['id', 'feature_by_volume']], on='id', how='left')

    return ret


def log_feature_prob(train, test, level=0):

    t = pd.merge(train[['id', 'fault_severity']], log_feature, on='id',
                 how='left').drop_duplicates()

    log_given_severity = \
       t.loc[t['fault_severity'] == level, 'log_feature'].value_counts() / \
       t['fault_severity'].value_counts()[level]

    severity_prob = \
        t['fault_severity'].value_counts()[level] / len(t)

    log_probs = \
        t['log_feature'].value_counts() / len(t)

    log_feature_probs = \
        (log_given_severity * severity_prob) / log_probs

    prob_df = pd.DataFrame({'probs': log_feature_probs})

    p = pd.merge(t, prob_df, left_on='log_feature', right_index=True, how='left')
    prob_table = pd.pivot_table(p, values='probs', index='id', columns='log_feature',
                                aggfunc=np.mean, fill_value=0)

    train = pd.merge(train, prob_table, left_on='id', right_index=True,
                     how='left')
    test = pd.merge(test, prob_table, left_on='id', right_index=True,
                    how='left')

    return train, test


def dangerous_log(train, test, level=0):

    feature_count = \
        train.loc[
            (train['fault_severity'] == level),
            ['log_feature', 'volume']
        ].groupby(by='log_feature', sort=False).sum()

    feature_total = \
        train.loc[
            (train['fault_severity'] == level),
            'volume'
        ].sum()

    danger = feature_count / feature_total

    train.loc[
        (train['fault_severity'] == level) & (train['log_feature'].isin(danger.index)),
            'dangerous_log'
    ] = danger['volume']

    test.loc[
        (test['fault_severity'] == level) & (test['log_feature'].isin(danger.index)),
            'dangerous_log'
    ] = danger['volume']

    return train, test


def danger_log(train, test):

    train, test = dangerous_log(train, test, level=0)
    train, test = dangerous_log(train, test, level=1)
    train, test = dangerous_log(train, test, level=2)

    return train, test


def base_features(df):

    df_event = event_type_features(df)
    df_resource = resource_type_features(df_event)
    df_severity = severity_type_features(df_resource)
    df_log = log_features(df_severity)

    df_log_count = log_feature_volume(df_log)
    df_severity_high = severity_high(df_log_count)
    df_resource_type = resource_type_event_count(df_severity_high)
    #df_resource_type_severity = resource_type_severity(df_resource_type)
    df_resource_type_log = resource_type_log(df_resource_type)
    #df_resource_type_log_sum = resource_type_log_sum(df_resource_type_log)
    #df_resource_type_log_total = resource_type_log_total(df_resource_type_log_sum)

    #df_severity_event = severity_event(df_resource_type_log_total)
    #df_severity_log = severity_log(df_severity_event)
    #df_severity_log_sum = severity_log_sum(df_severity_log)
    #df_event_resource_features = event_resource_features(df_severity_log)
    #df_event_log_features = event_log_features(df_event_resource_features)

    df_complete = df_resource_type_log

    return df_complete


def location_features(train, test, cutoff=0):

    train_locations = train[['location']]
    test_locations = test[['location']]

    train_locations.loc[:, 'train'] = True
    test_locations.loc[:, 'train'] = False

    train_distribution = train_locations['location'].value_counts()

    locations = pd.concat([train_locations, test_locations])
    locations.loc[:, 'location'] = locations['location'] \
        .apply(rare_category, args=(train_distribution, ),
               cutoff=cutoff, value='RareLocation')

    locations_bin = pd.get_dummies(locations['location'])
    locations_dummy = pd.concat([locations, locations_bin], axis=1)

    msk = locations_dummy['train']
    locations_dummy.drop(['train', 'location'], axis=1, inplace=True)

    train_locs = pd.concat([train, locations_dummy[msk]], axis=1)
    test_locs = pd.concat([test, locations_dummy[~msk]], axis=1)

    return train_locs, test_locs


def dangerous_location(train, test):

    danger = \
        (train.loc[(train['fault_severity'] == 2) | (train['fault_severity'] == 1), 'location']).value_counts() / \
        len(train.loc[(train['fault_severity'] == 2) | (train['fault_severity'] == 1), 'location'])

    msk = danger >= 0.025

    train.loc[train['location'].isin(danger[msk].index), 'dangerous'] = 1
    test.loc[test['location'].isin(danger[msk].index), 'dangerous'] = 1

    return train, test


def event_type_features(df):

    # d = event_type['event_type'].value_counts()
    # event_type.loc[:, 'event_type'] = event_type['event_type'] \
    #     .apply(filter_on_total, args=(d, ), total=1, value='RareEvent')

    event_type_dummies = pd.get_dummies(event_type['event_type'])
    event_dummy = pd.concat([event_type, event_type_dummies], axis=1)

    event_grpd = event_dummy \
        .groupby(event_dummy.id).sum()

    df_event = pd.merge(df, event_grpd, left_on='id', right_index=True)
    #df_event.loc[:, 'has_event_20'] = df_event['event_type 20']
    return df_event


def event_resource_features(df):

    e = pd.merge(event_type, resource_type, on='id')
    p = e.loc[e['event_type'] == 'event_type 23', ['id', 'resource_type']] \
        .groupby(by='id', as_index=False).count()
    df_23 = pd.merge(df, p, on='id', how='left')

    return df_23


def event_log_features(df):

    l = pd.merge(event_type, log_feature, on='id')
    p = l.loc[l['event_type'] == 'event_type 15', ['id', 'volume']] \
        .groupby(by='id', as_index=False).sum()

    df_15 = pd.merge(df, p[['id', 'volume']], on='id', how='left')

    p = l.loc[l['event_type'] == 'event_type 15', ['id', 'log_feature']] \
        .groupby(by='id', as_index=False).count()

    df_15c = pd.merge(df_15, p, on='id', how='left')


    p = l.loc[l['event_type'] == 'event_type 14', ['id', 'volume']] \
        .groupby(by='id', as_index=False).sum()
    df_14 = pd.merge(df_15c, p[['id', 'volume']], on='id', how='left')

    return df_14


def event_severity_prob(train, test, level=0):
    # print()
    # print(train.shape)
    # print(test.shape)
    # print()
    t = pd.merge(train[['id', 'fault_severity']], event_type, on='id',
                 how='left').drop_duplicates()

    event_given_severity = \
        t.loc[t['fault_severity'] == level, 'event_type'].value_counts() / \
        len(t.loc[t['fault_severity'] == level])

    severity_probs = \
        len(t.loc[t['fault_severity'] == level, :]) / len(t)

    event_probs = \
        t['event_type'].value_counts() / len(t['event_type'])

    event_severity_probs = \
        (event_given_severity * severity_probs) / event_probs
    # event_probs_test = \
    #     (event_given_severity * severity_probs) / event_probs
    prob_df = pd.DataFrame({'probs': event_severity_probs})

    p = pd.merge(t, prob_df, left_on='event_type', right_index=True, how='left')
    prob_table = pd.pivot_table(p, values='probs', index='id', columns='event_type',
                                aggfunc=np.median, fill_value=0)

    train = pd.merge(train, prob_table, left_on='id', right_index=True,
                     how='left')
    test = pd.merge(test, prob_table, left_on='id', right_index=True,
                    how='left')

    # grouped_event_probs = event_probs_df \
    #     .groupby(by=event_probs_df.index).median()

    # print()
    # print('grouped_event_probs')
    # print(grouped_event_probs)
    # print()

    # print()
    # print(train.shape)
    # print(test.shape)
    # print()
    # train = pd.merge(train, event_probs_df, left_on='event_type',
    #                  right_index=True, how='left')
    # test = pd.merge(test, event_probs_df, left_on='event_type',
    #                  right_index=True, how='left')
    # print()
    # print(train.shape)
    # print(test.shape)
    # print()

    return train, test


def event_severity(train, test):

    #event_grouped = event_type.groupby(by='event_type', as_index=False)

    # print()
    # print('event grouped')
    # print(event_grouped.head(10))
    # print()

    # print()
    # print(train.shape)
    # print(test.shape)

    # train = pd.merge(train, event_type, on='id', how='left')
    # test = pd.merge(test, event_type, on='id', how='left')

    # print()
    # print(train.shape)
    # print(test.shape)

    # train, test = event_severity_prob(train, test, level=0)
    # train, test = event_severity_prob(train, test, level=1)
    # train, test = event_severity_prob(train, test, level=2)
    #train, test = event_severity_prob(train, test, level=0)
    #train, test = event_severity_prob(train, test, level=1)
    train, test = event_severity_prob(train, test, level=2)

    #train = train.groupby(by=)
    return train, test


    # feature_count = \
    #     train.loc[
    #         (train['fault_severity'] == level),
    #         ['log_feature', 'volume']
    #     ].groupby(by='log_feature', sort=False).sum()

    # # event_given_severity = \
    # #     train
    # train = pd.merge(train, event_type, on='id')
    # test = pd.merge(test, event_type, on='id')

    # safe = \
    #     (train.loc[train['fault_severity'] == 0, 'event_type']).value_counts() / \
    #     len(train.loc[train['fault_severity'] == 0, 'event_type'])

    # msk = safe >= 0.01

    # train.loc[train['event_type'].isin(safe[msk].index), 'safe_event'] = 1
    # test.loc[test['event_type'].isin(safe[msk].index), 'safe_event'] = 1

    #return train, test


def rare_category(x, category_distribution, cutoff=1, value='Rare'):
    try:
        if category_distribution[x] < cutoff:
            return value
    except (ValueError, KeyError):
        return np.nan
    else:
        return x

