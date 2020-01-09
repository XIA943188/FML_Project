# -*- coding: UTF-8 -*-

import numpy as np
import warnings
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from data_processing import *


class VoteClassifier:
    def __init__(self, knn_n_neighbors=5, gb_n_estimators=100, rf_n_estimators='warn', rf_oob_score=False):
        self.algorithms = [
            KNeighborsClassifier(n_neighbors=knn_n_neighbors),
            GradientBoostingClassifier(n_estimators=gb_n_estimators, random_state=10),
            RandomForestClassifier(n_estimators=rf_n_estimators, oob_score=rf_oob_score, random_state=13)]

    def fit(self, X, y):
        for algorithm in self.algorithms:
            algorithm.fit(X, y)

    def predict(self, X):
        predictions = [algorithm.predict(X) for algorithm in self.algorithms]
        y = [max_count([y_pred[i] for y_pred in predictions]) for i in range(len(X))]
        return y


class Classifier:
    def __init__(self, n_label=4):
        self.n_label = n_label
        self.cluster = SpectralClustering(n_clusters=self.n_label, affinity='nearest_neighbors', gamma=10)
        self.labels = []
        self.target_labels = []

        for label1 in range(n_label):
            for label2 in range(n_label):
                if label1 < label2:
                    self.labels.append((label1, label2))

        self.voters = {label: VoteClassifier() for label in self.labels}
        self.compare = [KNeighborsClassifier(), GradientBoostingClassifier(), RandomForestClassifier()]
        print("Initialization of the model completed.")

    def fit(self, X_cluster, X_classifier, ratio=.2):
        self.cluster.fit(X_cluster)
        y = self.cluster.labels_.astype(np.int)
        self.target_labels = get_target_labels(X_cluster, y, self.n_label)
        X_train, X_test, y_train, y_test = divide_data(X_classifier, y, 1 - ratio)
        datas = separate(X_train, y_train)
        for label in self.labels:
            data, voter = datas[label], self.voters[label]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The default value of n_estimators will change",
                    category=FutureWarning)
                voter.fit(data[0], data[1])
        for algorithm in self.compare:
            algorithm.fit(X_train, y_train)
        yy = list(y)
        print("Training of the model completed.")
        print("# 0: %d, # 1: %d, # 2: %d, # 3: %d" % (yy.count(0), yy.count(1), yy.count(2), yy.count(3)))
        return X_test, y_test

    def predict(self, X_test, y_test):
        predictions = [self.voters[key].predict(X_test) for key in self.voters]
        y_pred = [max_count([y[i] for y in predictions]) for i in range(len(X_test))]
        y_pred_compare = [algorithm.predict(X_test) for algorithm in self.compare]
        print(self.target_labels)
        compare = [0, 0, 0]
        count, tot = 0, len(X_test)
        for i in range(tot):
            if y_test[i] == y_pred[i]:
                count += 1
            if y_pred_compare[0][i] == y_test[i]:
                compare[0] += 1
            if y_pred_compare[1][i] == y_test[i]:
                compare[1] += 1
            if y_pred_compare[2][i] == y_test[i]:
                compare[2] += 1
        accu = count / tot * 100
        accu_compare = [count_compare / tot * 100 for count_compare in compare]
        print("Prediction completed. # correct: %d, # prediction: %d, total accuracy is %.1f%%." % (count, tot, accu))
        print("The accuracy of the simple model:")
        print("\tKNeighborClassifier: %.1f%%\n\tGradientBoostingClassifier: %.1f%%\n\tRandomForestClassifier: %.1f%%"
              % (accu_compare[0], accu_compare[1], accu_compare[2]))
        return y_pred


if __name__ == '__main__':
    X_cluster, X_classifier = get_data('大盘数据.xlsx')
    classifier = Classifier()
    X_test, y_test = classifier.fit(X_cluster, X_classifier)
    y_pred = classifier.predict(X_test, y_test)




