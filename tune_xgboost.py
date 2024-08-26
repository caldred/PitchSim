import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_predict, cross_val_score

# Define the hyperparameter space for the XGBoost classifier
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
    'tree_method':      'hist',
    'device':           'cuda',
    'objective':        'binary:logistic',
    'nthread':          -1,
    'importance_type':  'total_gain',
    'validate_parameters': True,
    'eval_metric': 'logloss',
}

# Define the hyperparameter space for distilled models
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

def tune_xgboost(df: pd.DataFrame, target: str, features: list, 
                 param_dist: dict = classifier_params, 
                 model: type = XGBClassifier, 
                 scoring: str = 'neg_log_loss', 
                 max_evals: int = 30) -> xgb.XGBModel:
    """
    Perform hyperparameter tuning on XGBoost model and then train the model with the best parameters found.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - target (str): The name of the target column.
    - features (list): List of feature column names.
    - param_dist (dict): Dictionary of hyperparameters to be tuned. Default is `classifier_params`.
    - model (type): XGBoost model type (classifier or regressor). Default is `XGBClassifier`.
    - scoring (str): Scoring metric for evaluation. Default is `neg_log_loss`.
    - max_evals (int): Maximum number of evaluations for tuning. Default is 30.
    
    Returns:
    - XGBModel: A trained XGBoost model.
    """
    
    # Make a copy of the hyperparameter distribution
    xgb_params = param_dist.copy()

    # Split the dataset into training and test sets. Use the smaller of 80% of the data or 1,000,000 records for training.
    train_size = min(int(len(df) * 0.8), 1000000)
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], train_size=train_size, random_state=42)

    def xgb_eval(params: dict, X: pd.DataFrame = X_train, y: pd.Series = y_train) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Parameters:
        - params (dict): Dictionary of hyperparameters.
        - X (pd.DataFrame): Features dataframe for training.
        - y (pd.Series): Target series for training.
        
        Returns:
        - float: The negative mean of the cross-validation score.
        """
        
        # Convert some parameters to integers as required by XGBoost
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        
        # Initialize the model with the current parameters
        clf = model(**params)
        
        # Use 5-fold cross-validation to evaluate the model
        cv_splitter = KFold(n_splits=5, random_state=42, shuffle=True)
        score = -cross_val_score(clf, X, y, scoring=scoring, cv=cv_splitter).mean()
        return score
    
    # Perform hyperparameter optimization using the Tree-structured Parzen Estimator (TPE) method
    trials = Trials()
    best = fmin(fn=xgb_eval, space=xgb_params, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    # Update the hyperparameters with the best values found
    for item in best:
        xgb_params[item] = best[item]

    # Convert to integers as required
    xgb_params['max_depth'] = int(xgb_params['max_depth'])
    xgb_params['min_child_weight'] = int(xgb_params['min_child_weight'])
    
    # Adjust the learning rate and number of estimators for final training
    xgb_params['learning_rate'] *= 0.1
    xgb_params['n_estimators'] *= 10

    # Train the model with the optimized hyperparameters on the entire dataset
    xgb_model = model(**xgb_params)
    xgb_model.fit(df[features], df[target])

    return xgb_model