import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
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


train_op = pd.read_csv('../train/op.csv')
train_trans = pd.read_csv('../train/trans.csv')
train_label = pd.read_csv('../train/label.csv')
sub_op = pd.read_csv('../original_data/operation_round1_new.csv')
sub_trans = pd.read_csv('../original_data/transaction_round1_new.csv')
sub_label = pd.read_csv('../original_data/sub.csv')
y = train_label['Tag']


def get_math_feature(data, label, feature):
    label = label.merge(data.groupby(['UID'])[feature].count().reset_index(), on='UID',
                        how='left')
    label = label.merge(data.groupby(['UID'])[feature].nunique().rename(feature + '_nunique').reset_index(), on='UID',
                        how='left')
    label = label.merge(data.groupby(['UID'])[feature].max().rename(feature + '_max').reset_index(), on='UID',
                        how='left')
    label = label.merge(data.groupby(['UID'])[feature].min().rename(feature + '_min').reset_index(), on='UID',
                        how='left')
    label = label.merge(data.groupby(['UID'])[feature].sum().rename(feature + '_sum').reset_index(), on='UID',
                        how='left')
    label = label.merge(data.groupby(['UID'])[feature].mean().rename(feature + '_mean').reset_index(), on='UID',
                        how='left')
    label = label.merge(data.groupby(['UID'])[feature].std().rename(feature + '_std').reset_index(), on='UID',
                        how='left')
    label = label.merge(data.groupby(['UID'])[feature].median().rename(feature + '_median').reset_index(), on='UID',
                        how='left')
    return label


def get_all_math_feature(data, label, feature):
    label = get_math_feature(data, label, feature)
    label = label.merge(data.groupby(['UID'])[feature].quantile(0.75).rename(feature + '_3/4').reset_index(), on='UID',
                        how='left')
    label = label.merge(data.groupby(['UID'])[feature].quantile(0.25).rename(feature + '_1/4').reset_index(), on='UID',
                        how='left')
    label = label.merge(data.groupby(['UID'])[feature].agg(lambda x: np.mean(pd.Series.mode(x))).rename(
        feature + '_mode').reset_index(), on='UID', how='left')
    return label


def get_object_feature(data, label, feature):
    label = label.merge(
        data.groupby(['UID'])[feature].count().rename(feature + '_count').reset_index(),
        on='UID', how='left')
    label = label.merge(
        data.groupby(['UID'])[feature].nunique().rename(feature + '_nunique').reset_index(),
        on='UID', how='left')

    return label


def get_base_feature(op, trans, label):
    for feature in op.columns[1:]:
        if feature in ['day', 'time', 'ip2', 'ip2_sub']:
            continue
        if op[feature].dtype == 'object':
            label = get_object_feature(op, label, feature)
        else:
            label = get_math_feature(op, label, feature)
    for feature in trans.columns[1:]:
        if feature in ['day', 'time', 'code1', 'code2']:
            continue
        if trans[feature].dtype == 'object':
            label = get_object_feature(trans, label, feature)
        else:
            label = get_math_feature(trans, label, feature)
    return label


# base feature
train_feature_base = get_base_feature(train_op, train_trans, train_label)
sub_feature_base = get_base_feature(sub_op, sub_trans, sub_label)
print('get base feature', train_feature_base.shape[1], sub_feature_base.shape[1])


# time feature
def get_time(data, label, feature='time'):
    temp = pd.DataFrame()
    temp[feature] = data[feature].apply(
        lambda x: int(x.split(':')[0]) + int(x.split(':')[1]) / 60 + int(x.split(':')[2]) / 3600)
    temp['UID'] = data['UID']
    label = get_all_math_feature(data, label, feature)
    label = label.merge((temp.groupby(['UID'])[feature].max() - temp.groupby(['UID'])[feature].min())
                        .rename(feature + '_max-min').reset_index(), on='UID', how='left')
    return label


def get_time_feature(tlabel, slabel):
    train_feature = get_time(train_op, tlabel)
    train_feature = get_time(train_trans, train_feature)

    sub_feature = get_time(sub_op, slabel)
    sub_feature = get_time(sub_trans, sub_feature)
    return train_feature, sub_feature


train_feature_time, sub_feature_time = get_time_feature(train_feature_base, sub_feature_base)
print('get feature', train_feature_time.shape[1], sub_feature_time.shape[1])


def get_list(train_data, sub_data, feature, num):
    train_tag_list = list(train_data[feature].unique())
    test_tag_list = list(sub_data[feature].unique())
    tag_list = list(set(train_tag_list) & set(test_tag_list))
    black_id = train_label.loc[train_label['Tag'] == 1, ['UID']]["UID"]
    black_index = train_data['UID'].isin(black_id)
    black_data = train_data[black_index][['UID', feature]]
    need_list = black_data[feature].value_counts()[black_data[feature].value_counts().index.isin(tag_list)].index[:num]
    return need_list


def tag_select(data, label, feature, itype, num):
    if itype == 'op':
        tag_list = get_list(train_op, sub_op, feature, num)
    else:
        tag_list = get_list(train_trans, sub_trans, feature, num)
    for tag in tag_list:
        label = label.merge(data.loc[data[feature] == tag].groupby(['UID']).size().rename(
            feature + '_' + str(tag) + '_count').reset_index()
                            , on='UID', how='left')

    return label


# mode select feature
train_feature_mode = tag_select(train_op, train_feature_time, feature='mode', itype='op', num=40)
sub_feature_mode = tag_select(sub_op, sub_feature_time, feature='mode', itype='op', num=40)
print('get feature', train_feature_mode.shape[1], sub_feature_mode.shape[1])
# merchant select feature
train_feature_merchant = tag_select(train_trans, train_feature_mode, feature='merchant', itype='trans', num=40)
sub_feature_merchant = tag_select(sub_trans, sub_feature_mode, feature='merchant', itype='trans', num=40)
print('get feature', train_feature_merchant.shape[1], sub_feature_merchant.shape[1])
# geo_code select feature
train_feature_geo = tag_select(train_trans, train_feature_time, feature='geo_code', itype='trans', num=40)
sub_feature_geo = tag_select(sub_trans, sub_feature_time, feature='geo_code', itype='trans', num=40)
train_feature_geo = tag_select(train_op, train_feature_geo, feature='geo_code', itype='op', num=40)
sub_feature_geo = tag_select(sub_op, sub_feature_geo, feature='geo_code', itype='op', num=40)
print('get feature', train_feature_geo.shape[1], sub_feature_geo.shape[1])

# merge data
trans_columns = list(train_trans.columns)
op_columns = list(train_op.columns)
to_columns = list(set(trans_columns) & set(op_columns))
train_trans1 = train_trans[to_columns]
train_op1 = train_op[to_columns]
merge_train_data = train_trans1.append(train_op1)
sub_trans1 = sub_trans[to_columns]
sub_op1 = sub_op[to_columns]
merge_sub_data = sub_trans1.append(sub_op1)


# merge base feature
def merge_base_feature(data, label):
    for feature in data.columns:
        print(feature)
        if feature == 'UID':
            continue
        if data[feature].dtype == 'object':
            label = get_object_feature(data, label, feature)
        else:
            label = get_all_math_feature(data, label, feature)

    return label


train_feature_mergebase = merge_base_feature(merge_train_data, train_feature_geo)
sub_feature_mergebase = merge_base_feature(merge_sub_data, sub_feature_geo)
print('get feature', train_feature_mergebase.shape[1], sub_feature_mergebase.shape[1])


# merge mac1 feature
def merge_mac1_select(data, label, feature, num):
    tag_list = get_list(merge_train_data, merge_sub_data, feature, num)
    for tag in tag_list:
        label = label.merge(data.loc[data[feature] == tag].groupby(['UID']).size().rename(
            feature + '_' + str(tag) + '_count').reset_index()
                            , on='UID', how='left')

    return label


train_feature_mergemac1 = merge_mac1_select(merge_train_data, train_feature_mergebase, feature='mac1', num=40)
sub_feature_mergemac1 = merge_mac1_select(merge_sub_data, sub_feature_mergebase, feature='mac1', num=40)
print('get feature', train_feature_mergemac1.shape[1], sub_feature_mergemac1.shape[1])

# train
train_feature = train_feature_mergemac1
sub_feature = sub_feature_mergemac1
train_feature = train_feature.drop(['Tag', 'UID'], axis=1).fillna(-1)
sub_feature = sub_feature.drop(['Tag', 'UID'], axis=1).fillna(-1)
print('get feature', len(train_feature.columns), len(sub_feature.columns))
lgb_model = lgb.LGBMClassifier(n_estimators=10000, objective='binary',
                               class_weight='balanced', learning_rate=0.05,
                               reg_alpha=.1, reg_lambda=.1,
                               subsample=1, n_jobs=-1, random_state=50)
skf = StratifiedKFold(n_splits=5, random_state=50, shuffle=True)
valid_0_best_loss = []
valid_1_best_loss = []
train_preds = np.zeros(train_feature.shape[0])
sub_preds = np.zeros(sub_feature.shape[0])

for index, (train_index, test_index) in enumerate(skf.split(train_feature, y)):
    lgb_model.fit(train_feature.iloc[train_index], y.iloc[train_index], verbose=50,
                  eval_set=[(train_feature.iloc[train_index], y.iloc[train_index]),
                            (train_feature.iloc[test_index], y.iloc[test_index])], early_stopping_rounds=30)
    valid_0_best_loss.append(lgb_model.best_score_['valid_0']['binary_logloss'])
    valid_1_best_loss.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(valid_0_best_loss)
    print(valid_1_best_loss)
    train_preds[test_index] = lgb_model.predict_proba(train_feature.iloc[test_index],
                                                      num_iteration=lgb_model.best_iteration_)[:, 1]
    sub_pred = lgb_model.predict_proba(sub_feature, num_iteration=lgb_model.best_iteration_)[:, 1]
    sub_preds += sub_pred / 5

m = tpr_weight_funtion(y_predict=train_preds, y_true=y)
print('valid_0_best_score:', np.mean(valid_0_best_loss))
print('valid_1_best_score', np.mean(valid_1_best_loss))
print('score:', m)
sub_label['Tag'] = sub_preds
sub_label.to_csv('../sub/sub_loss/baseline.csv', index=False)

# feature importance
column_importance = pd.DataFrame({'fea': list(train_feature.columns), 'imp': lgb_model.feature_importances_})
remove_col = []
for idx in range(len(column_importance)):
    col = column_importance['fea'][idx]
    if col == 'UID':
        continue
    wgt = column_importance['imp'][idx]
    if wgt <= 0:
        if col in train_feature.columns:
            remove_col.append(col)

            train = train_feature.drop(col, axis=1)
            sub_feature = sub_feature.drop(col, axis=1)
print(train_feature.shape), print(sub_feature.shape)


# rule
def find_black(data, feature):
    black = (data.groupby([feature])['Tag'].sum() / data.groupby([feature])['Tag'].count()).sort_values(
        ascending=False)
    tag_count = data.groupby([feature])['Tag'].count().reset_index()
    black = black.reset_index().merge(tag_count, on=feature, how='left')
    black = black.sort_values(by=['Tag_x', 'Tag_y'], ascending=False)
    return black


train_trans = train_trans.merge(train_label, on='UID', how='left')
black = find_black(train_trans, 'merchant')
rule_code = black.sort_values(['Tag_x', 'Tag_y'], ascending=False).iloc[:50].merchant.tolist()
test_rule_uid = pd.DataFrame(sub_trans[sub_trans['merchant'].isin(rule_code)].UID.unique())
pred_data_rule = sub_label.merge(test_rule_uid, left_on='UID', right_on=0, how='left')
pred_data_rule['Tag'][(pred_data_rule[0] > 0)] = 1
pred_data_rule[['UID', 'Tag']].to_csv('subrule.csv', index=False)
