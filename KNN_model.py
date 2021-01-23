import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import decomposition
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from math import sqrt
import random


dataset = pd.read_table('data/movies_data_12-18.csv', sep=';')

def append_feature(X, feature, dataset):
    feature_set = list(set(dataset.iloc[:, feature].values))
    feature_org = dataset.iloc[:, feature].values
    feature_con = []
    for entry in feature_org:
        feature_con.append([feature_set.index(entry)])
    X = np.append(X, feature_con, 1)
    return X

def run_knn(dataset, features, neighbors, appendable_features = [], enable_scaler = False, PCA = 0):

    X = dataset.iloc[:, features].values
    y = dataset.iloc[:, 7].values

    for f in appendable_features:
        X = append_feature(X, f, dataset)

    if enable_scaler:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    if PCA != 0:
        pca = decomposition.PCA(n_components=PCA)
        X = pca.fit_transform(X)

    info = []
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    kf = KFold(n_splits=10, shuffle=True)
    for train, test in kf.split(X):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]

        model = KNeighborsRegressor(n_neighbors=neighbors)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)

        info_row = [mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), sqrt(mean_squared_error(y_test, y_pred)), mean_squared_error(y_train, y_pred_train), r2_score(y_train, y_pred_train), sqrt(mean_squared_error(y_train, y_pred_train))]
        info.append(info_row)

    stats = [0, 0, 0, 0, 0, 0]
    for i in range(10):
        for j in range(6):
            stats[j] += info[i][j]

    for j in range(6):
        stats[j] /= 10
    return stats

stats = run_knn(dataset, [6, 11, 12], 10, [3], True)
print(stats)