import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import random


dataset = pd.read_table('data/movies_data_12-18.csv', sep=';')

#Converts categorical feature to numerical and appends to X
def append_feature(X, feature, dataset):
    feature_set = list(set(dataset.iloc[:, feature].values))
    feature_org = dataset.iloc[:, feature].values
    feature_con = []
    for entry in feature_org:
        feature_con.append([feature_set.index(entry)])
    X = np.append(X, feature_con, 1)
    return X

#Used to get results, this is for knn, but feel free to use parts of it
#appendable_features is for categorical features, scaler is for data normalization, PCA is for features selection
def evaluate_knn(dataset, features, neighbors, appendable_features = [], enable_scaler = False, PCA = 0, seed = 5):
    #id;tconst;title;genres;directors;writers;averageRating;boxOffice;yearOfBoxOffice;year;directorsNames;votes;budget
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    classifier = KNeighborsRegressor(n_neighbors=neighbors)
    classifier.fit(X_train, y_train)

    y_pred_train = classifier.predict(X_train)
    y_pred = classifier.predict(X_test)

    info = [mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error(y_train, y_pred_train), r2_score(y_train, y_pred_train)]
    return info

max_info = [0,0,0,0]
max_neighbors = 1
max_features = [0]
max_ap_feat = [0]
max_en_scaler = False
max_PCA = 0

#id;tconst;title;genres;directors;writers;averageRating;boxOffice;yearOfBoxOffice;year;directorsNames;votes;budget
#Those are features that are reasonable to use
num_features = [6,8,9,11,12]
cat_features = [3,4,5]

def add_features(feat, sel, r):
    f = []
    for j in range(0, r):
        if sel[j] == 1:
            f.append(feat[j])
    return f

inputs = []

#This loops are to use all possible combinations
for i in itertools.product([0,1],repeat=5):
    features = add_features(num_features, i, 5)

    if features == []:
        features = [6]

    for j in itertools.product([0, 1], repeat=3):
        ap_features = add_features(cat_features, j, 3)

        for k in range(2):
            en_scaler = k

            for l in range(0, len(features) + len(ap_features)):
                PCA = l

                for z in range(1, 20):
                    input = [features, ap_features, en_scaler, PCA, z, 0, 0, 0, 0]
                    inputs.append(input)


#This loop is for runing tests with all possible combinations with 20 different splits of data
for i in range(20):
    seed = random.randint(1, 1000)

    for id in range(len(inputs)):
        info = evaluate_knn(dataset, inputs[id][0], inputs[id][4], inputs[id][1], inputs[id][2], inputs[id][3], seed)
        print(str(id) + " " + str(info))
        inputs[id][5] += info[0]
        inputs[id][6] += info[1]
        inputs[id][7] += info[2]
        inputs[id][8] += info[3]


inputs = sorted(inputs, key=lambda x: x[6], reverse=True)
print("--------------------------------")
for i in range(20):
    print(str(inputs[i]) + " " + str(inputs[i][5] / 20) + " " + str(inputs[i][6] / 20) + " " + str(inputs[i][7] / 20) + " " + str(inputs[i][8] / 20))

