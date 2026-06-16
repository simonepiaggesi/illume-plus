from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets, svm
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid, ParameterSampler, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, spearmanr
from numpy.random import default_rng
from collections import Counter
from itertools import groupby
from sklearn.cluster import KMeans
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

import optuna
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline


def lgbm_optuna_eval(
    X_train, y_train,
    X_val, y_val,
    f1_average="macro",
    n_trials=50,
    random_state=42,
    n_jobs=4
):
    n_classes = len(np.unique(y_train))

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    def objective(trial):

        params = {
            # -------- Core boosting --------
            "n_estimators": 2000,               
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.05, log=True
            ),

            # -------- Tree complexity --------
            "num_leaves": trial.suggest_int("num_leaves", 4, 16),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 50, 200
            ),

            # -------- Regularization --------
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1.0, 50.0),
            "min_gain_to_split": trial.suggest_float(
                "min_gain_to_split", 0.0, 5.0
            ),

            # -------- Subsampling --------
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.01, 0.10
            ),
            "feature_fraction_bynode": trial.suggest_float(
                "feature_fraction_bynode", 0.5, 1.0
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.6, 0.9
            ),
            "bagging_freq": 1,
        }

        clf = LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
            early_stopping_rounds=100,
            **params
        )

        scores = []

        for tr_idx, va_idx in cv.split(X_train, y_train):
            clf.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[va_idx], y_train[va_idx])],
                eval_metric="multi_logloss",
            )

            preds = clf.predict(X_train[va_idx])
            scores.append(
                f1_score(
                    y_train[va_idx],
                    preds,
                    average=f1_average
                )
            )

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")

    pbar = tqdm(total=n_trials, desc="Optuna CV trials")
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[lambda s, t: pbar.update(1)]
    )
    pbar.close()

    best_clf = LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
        early_stopping_rounds=100,
        **study.best_params
    )

    best_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],   
        eval_metric="multi_logloss"
    )

    val_f1 = f1_score(
        y_val,
        best_clf.predict(X_val),
        average=f1_average
    )

    return best_clf, val_f1


def tree_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
                    param_grid = {'min_samples_split': [2, 0.002, 0.01, 0.05, 0.1, 0.2],
                                  'min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
                                  'max_depth': [None, 2, 4, 6, 8, 10, 12, 16]}, 
                    n_jobs=-1):

    param_list = list(ParameterGrid(param_grid))

    acc = []
    for params in param_list:

        clf = DecisionTreeClassifier(random_state=42, **params)
        clf.fit(Z_train, Y_train)
        Y_pred = clf.predict(Z_test)
        acc.append(f1_score(Y_test, Y_pred, average=f1_average))

    best_params = param_list[np.argmax(acc)]
    best_clf = DecisionTreeClassifier(random_state=42, **best_params)
    best_clf.fit(Z_train, Y_train)
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)    


def linear_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
                    param_grid = {'penalty' : ['l1', 'l2'],
                                  'C': [0.001, 0.01, 0.05, 0.1, 1., 10.],
                                  'max_iter' : [100, 1000, 2000, 5000]}, 
                    n_jobs=-1):

    param_list = list(ParameterGrid(param_grid))

    acc = []
    for params in param_list:

        clf = LogisticRegression(solver='liblinear',random_state=42, **params)
        clf.fit(Z_train, Y_train)
        Y_pred = clf.predict(Z_test)
        acc.append(f1_score(Y_test, Y_pred, average=f1_average))

    best_params = param_list[np.argmax(acc)]
    best_clf = LogisticRegression(solver='liblinear',random_state=42, **best_params)
    best_clf.fit(Z_train, Y_train)
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)    
