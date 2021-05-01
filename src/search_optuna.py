import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn import model_selection

from functools import partial
import optuna

from . import regression_metrics

def optimize(trial, df):

    n_estimators = trial.suggest_int("n_estimators", 50, 1000)
    num_leaves = trial.suggest_int("num_leaves", 10, 500)
    learning_rate = trial.suggest_uniform("learning_rate", 0.01, 1.0)

    modely = lgb.LGBMRegressor(
        n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate)

    modelx = lgb.LGBMRegressor(
        n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate)

    modelf = lgb.LGBMClassifier(
        n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate)

    kf = model_selection.KFold(n_splits=5)
    accuracies = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):       
        
        df_train = df.loc[train_idx]
        df_val = df.loc[val_idx]
        
        x_train = df_train.iloc[:,:-5]
        y_trainx = df_train.iloc[:,-5]
        y_trainy = df_train.iloc[:,-4]
        y_trainf = df_train.iloc[:,-3]        
        
        x_val = df_val.iloc[:,:-5]
        y_valx = df_val.iloc[:,-5]
        y_valy = df_val.iloc[:,-4]
        y_valf = df_val.iloc[:,-3]          

        modelx.fit(x_train, y_trainx)
        modely.fit(x_train, y_trainy)
        modelf.fit(x_train, y_trainf)

        test_predsx = modelx.predict(x_val)
        test_predsy = modely.predict(x_val)
        test_predsf = modelf.predict(x_val)
  
        fold_metric = regression_metrics.iln_comp_metric(test_predsx, test_predsy, test_predsf, y_valx, y_valy, y_valf)
        comp_metric.append(fold_metric)
    return np.mean(comp_metric)


def hyperpara_search_optuna(df):

    optimization_function = partial(optimize, df=df)

    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_function, n_trials=15)
