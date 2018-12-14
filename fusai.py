import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold


def tpr_weight_funtion(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer - 0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer - 0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer - 0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3


# sum could be replace by any one of ['mean','max','min','std','median']
# but don't konw whether model work well

def get_feature(op, trans, label):
    for feature in op.columns:
        if feature in ['day', 'ip2', 'ip2_sub']:
            continue
        if feature != 'UID':
            label = label.merge(op.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')
            # 'device1','device2','device_code1','device_code2','device_code3','mac1','mac2','ip1','ip1_sub','wifi'
            # 'mode','version','os','success'
        for deliver in ['ip1', 'ip1_sub', 'wifi', 'mac1', 'mac2', 'geo_code', 'mode', 'device1', 'device2',
                        'device_code1', 'device_code2', 'device_code3']:
            if feature not in deliver:
                if feature != 'UID':
                    temp = op[['UID', deliver]].merge(op.groupby([deliver])[feature].count().reset_index(), on=deliver,
                                                      how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(), on=deliver,
                                                   how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                else:
                    temp = op[['UID', deliver]].merge(op.groupby([deliver])[feature].count().reset_index(), on=deliver,
                                                      how='left')[['UID_x', 'UID_y']]
                    temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(), on=deliver,
                                                   how='left')[['UID_x', 'UID_y']]
                    temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp

    for feature in trans.columns:
        if feature in ['day', 'code1', 'code2']:
            continue
        if feature not in ['trans_amt', 'bal']:
            if feature != 'UID':
                label = label.merge(trans.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
                label = label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')
            # 'device1','device2','device_code1','device_code2','device_code3','mac1','mac2','ip1','ip1_sub','acc_id1','acc_id2','acc_id1'
            # 'merchant','channel','amt_src1','amt_src2','trans_type1','trans_type2','market_type'

            for deliver in ['merchant', 'ip1', 'mac1', 'geo_code', 'device1', 'device2', 'device_code1',
                            'device_code2', 'device_code3', 'ip1_sub', 'acc_id1', 'acc_id2', 'acc_id1', 'amt_src2']:
                if feature not in deliver:
                    if feature != 'UID':
                        temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),
                                                             on=deliver, how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
                        temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),
                                                             on=deliver, how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
                    else:
                        temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),
                                                             on=deliver, how='left')[['UID_x', 'UID_y']]
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
                        temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),
                                                             on=deliver, how='left')[['UID_x', 'UID_y']]
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
    print("Done")
    return label


train_op = pd.read_csv('../train/op_black.csv')
train_trans = pd.read_csv('../train/trans_black.csv')
train_label = pd.read_csv('../train/sub_black.csv')
sub_op = pd.read_csv('../sub/sub_data/op.csv')
sub_trans = pd.read_csv('../sub/sub_data/trans.csv')
sub_label = pd.read_csv('../sub/sub_data/sub.csv')
label = train_label['Tag']

base_train = get_feature(train_op, train_trans, train_label).fillna(-1)
base_test = get_feature(sub_op, sub_trans, sub_label).fillna(-1)
print('get base fearture!')

# merge all feature
# these are get by function in chusai.py
train = base_train
test = base_test
num_base = len(train.columns)
name_list = ['mode_select_feature.csv', 'merchant_select_feature.csv', 'geo_code_select_feature.csv',
             'merge_mac1_select_feature.csv']
for name in name_list:
    print(name)
    train = train.merge(pd.read_csv('../feature/train/' + name), on='UID', how='left')
    test = test.merge(pd.read_csv('../feature/sub/' + name), on='UID', how='left')
train = train.drop(['UID', 'Tag'], axis=1).fillna(-1)
test = test.drop(['UID', 'Tag'], axis=1).fillna(-1)
print('get feature', len(train.columns), len(test.columns))
# Absolute value correlation matrix
corr_matrix = train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


def drop_importance(train, test, impo):
    column_importance = pd.DataFrame({'fea': list(train.columns), 'imp': impo})
    remove_col = []
    for idx in range(len(column_importance)):
        col = column_importance['fea'][idx]
        wgt = column_importance['imp'][idx]
        if wgt <= 0:
            if col in train.columns:
                remove_col.append(col)

                train = train.drop(col, axis=1)
                test = test.drop(col, axis=1)
    return train, test


def drop_feature(threshold_base, threshold_other, train, test, upper):
    to_drop = []
    for column in upper.columns[:num_base]:
        if any(upper[column] >= threshold_base):
            to_drop.append(column)
    for column in upper.columns[num_base:]:
        if any(upper[column] >= threshold_other):
            to_drop.append(column)
    print('There are %d columns to remove.' % (len(to_drop)))
    return train.drop(columns=to_drop), test.drop(columns=to_drop)


def pipeline(iteration, random_seed, threshold_base, threshold_other, train, test, upper, learning_rate):
    i = 0
    train_feature, test_feature = drop_feature(threshold_base, threshold_other, train, test, upper)
    print(train_feature.shape, test_feature.shape)
    skf = StratifiedKFold(n_splits=5, random_state=50, shuffle=True)
    while True:
        lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, reg_alpha=3, reg_lambda=5, max_depth=-1,
                                       n_estimators=5000, objective='binary', subsample=0.9, colsample_bytree=0.77,
                                       subsample_freq=1, learning_rate=learning_rate,
                                       random_state=random_seed, n_jobs=16, min_child_weight=4, min_child_samples=5,
                                       min_split_gain=0)
        train_preds = np.zeros(train_feature.shape[0])
        sub_preds = np.zeros(test_feature.shape[0])

        for index, (train_index, test_index) in enumerate(skf.split(train_feature, label)):
            lgb_model.fit(train_feature.iloc[train_index], label.iloc[train_index], verbose=50,
                          eval_set=[(train_feature.iloc[train_index], label.iloc[train_index]),
                                    (train_feature.iloc[test_index], label.iloc[test_index])], early_stopping_rounds=30)
            train_preds[test_index] = lgb_model.predict_proba(train_feature.iloc[test_index],
                                                              num_iteration=lgb_model.best_iteration_)[:,
                                      1]
            test_pred = lgb_model.predict_proba(test_feature, num_iteration=lgb_model.best_iteration_)[:, 1]
            sub_preds += test_pred / 5
        m = tpr_weight_funtion(y_predict=train_preds, y_true=label)
        print('tpr score:', m)
        train_feature, test_feature = drop_importance(train_feature, test_feature, lgb_model.feature_importances_)
        if i != 0:
            sub_label['Tag'] = sub_preds
            sub_label.to_csv('../lgb_30/lgb{0}.csv'.format(iteration), index=False, encoding='utf-8')
            break
        i = 1

# 30 model fusion
import random

# max_depth = [4,5,6]
# reg_alpha = [i/100.0 for i in range(10,50)]
# reg_lambda = [i/100.0 for i in range(10,50)]
# subsample = [i/1000.0 for i in range(750,850)]
# colsample_bytree = [i/1000.0 for i in range(950,1000)]
random_seed = list(range(2018))
threshold_base = [i / 10000.0 for i in range(9850, 9950)]
threshold_other = [i / 1000.0 for i in range(400, 600)]
learning_rate = [i / 10000.0 for i in range(400, 600)]
random.shuffle(random_seed)
random.shuffle(threshold_base)
random.shuffle(threshold_other)
random.shuffle(learning_rate)
# random.shuffle(max_depth)
# random.shuffle(reg_alpha)
# random.shuffle(reg_lambda)
# random.shuffle(subsample)
# random.shuffle(colsample_bytree)
for i in range(30):
    print('iter:', i)
    pipeline(i, random_seed[i], threshold_base[i], threshold_other[i], train, test, upper, learning_rate[i])

import os

path = 'lgb_30/'
name_list = list(path + name for name in os.listdir(path))
sub = pd.read_csv(name_list[0])
sub['Tag'] = sub['Tag'] / 30
for name in name_list[1:]:
    sub['Tag'] += pd.read_csv(name)['Tag'] / 30
sub.to_csv('lgb_30_sub.csv', index=False)
