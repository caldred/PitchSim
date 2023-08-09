import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

classifier_params = {
    'learning_rate':    hp.lognormal('learning_rate', -2.4, 0.3),
    'max_depth':        hp.quniform('max_depth', 3, 12, 1),
    'min_child_weight': hp.qloguniform('min_child_weight', 2, 7, 1),
    'gamma':            hp.lognormal('gamma', 1, 2),
    'colsample_bytree': 1,
    'colsample_bylevel':1,
    'colsample_bynode': 1,
    'subsample':        0.2,
    'reg_alpha':        0,
    'reg_lambda':       1,
    'n_estimators':     100,
    'sampling_method':  'gradient_based',
    'tree_method':      'gpu_hist',
    'gpu_id':           0,
    'objective':        'binary:logistic',
    'nthread':          -1,
    'importance_type':  'total_gain',
    'validate_parameters': True,
    'eval_metric': 'logloss',
}

distill_params = {
    'learning_rate':    0.1,
    'max_depth':        15,
    'min_child_weight': hp.qloguniform('min_child_weight', 1, 4, 1),
    'gamma':            0,
    'colsample_bytree': 1,
    'colsample_bylevel':1,
    'colsample_bynode': 1,
    'subsample':        0.8,
    'reg_alpha':        0,
    'reg_lambda':       1,
    'n_estimators':     100,
    'sampling_method':  'uniform',
    'tree_method':      'hist',
    'gpu_id':           0,
    'objective':        'reg:squarederror',
    'nthread':          10,
    'importance_type':  'total_gain',
    'validate_parameters': True,
    'eval_metric': 'rmse',
}

def tune_xgboost(df, target, features, param_dist=classifier_params, model=XGBClassifier, scoring='neg_log_loss', max_evals=30):

    xgb_params = param_dist.copy()

    train_size = min(int(len(df)*0.8), 1000000)

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], train_size=train_size, random_state=42)

    def xgb_eval(params, X=X_train, y=y_train):
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        clf = model(**params)
        cv_splitter = KFold(n_splits=5, random_state=42, shuffle=True)
        score = -cross_val_score(clf, X, y, scoring=scoring, cv=cv_splitter).mean()
        return score
    
    trials = Trials()
    best = fmin(fn=xgb_eval, space=xgb_params, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    for item in best:
        xgb_params[item] = best[item]

    xgb_params['max_depth'] = int(xgb_params['max_depth'])
    xgb_params['min_child_weight'] = int(xgb_params['min_child_weight'])
    xgb_params['learning_rate'] *= 0.1
    xgb_params['n_estimators'] *= 10

    xgb_model = model(**xgb_params)
    xgb_model.fit(df[features], df[target])

    return xgb_model


