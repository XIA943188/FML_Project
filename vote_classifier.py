# -*- coding: UTF-8 -*-

import numpy as np
import warnings
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from data_processing import *


class VoteClassifier:
    def __init__(self, knn_n_neighbors=5, gb_n_estimators=400, rf_n_estimators='warn', rf_oob_score=False):
        self.algorithms = [
            (KNeighborsClassifier(n_neighbors=knn_n_neighbors), 1.),
            (GradientBoostingClassifier(n_estimators=gb_n_estimators, random_state=10), 10.),
            (RandomForestClassifier(n_estimators=rf_n_estimators, oob_score=rf_oob_score, random_state=13), 1.)]
        self.weights = []
        self.labels = []

    def weighted_voting(self, y_preds, weights):
        y_pred = []
        for i in range(len(y_preds[0])):
            prediction = 0.
            for j in range(len(weights)):
                prediction += weights[j] * y_preds[j][i]
            if abs(prediction - self.labels[0]) < abs(prediction - self.labels[1]):
                y_pred.append(self.labels[0])
            elif abs(prediction - self.labels[0]) > abs(prediction - self.labels[1]):
                y_pred.append(self.labels[1])
            else:
                y_pred.append(y_vote[i])
        return y_pred

    def fit(self, X, y):
        self.labels = list(set(y))
        y_preds, weights = [], []
        for algorithm, weight in self.algorithms:
            algorithm.fit(X, y)
            y_pred = algorithm.predict(X)
            y_preds.append(y_pred)
            weights.append(accuracy_calc(y, y_pred) * weight)
        y_vote = [max_count([y_pred[i] for y_pred in y_preds]) for i in range(len(X))]
        y_preds.append(y_vote)
        weights.append(accuracy_calc(y, y_vote))
        tot_weight = 0.
        for weight in weights:
            tot_weight += weight
        self.weights = [weight/tot_weight for weight in weights]

    def predict(self, X):
        y_preds = [algorithm.predict(X) for algorithm, weight in self.algorithms]
        y_vote = [max_count([y_pred[i] for y_pred in y_preds]) for i in range(len(X))]
        y_preds.append(y_vote)
        y = self.weighted_voting(y_preds, self.weights)
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
        print("Target label: ", self.target_labels)
        compare, target_compare = [0, 0, 0], [0, 0, 0]
        count, target_count, tot, target_tot = 0, 0, len(X_test), 0
        for i in range(tot):
            if y_test[i] in self.target_labels:
                target_tot += 1
            if y_test[i] == y_pred[i]:
                count += 1
                if y_test[i] in self.target_labels:
                    target_count += 1
            if y_pred_compare[0][i] == y_test[i]:
                compare[0] += 1
                if y_test[i] in self.target_labels:
                    target_compare[0] += 1
            if y_pred_compare[1][i] == y_test[i]:
                compare[1] += 1
                if y_test[i] in self.target_labels:
                    target_compare[1] += 1
            if y_pred_compare[2][i] == y_test[i]:
                compare[2] += 1
                if y_test[i] in self.target_labels:
                    target_compare[2] += 1
        accu, target_accu = count / tot * 100, target_count / target_tot * 100
        accu_compare = [count_compare / tot * 100 for count_compare in compare]
        accu_target_compare = [count_target_compare / target_tot * 100 for count_target_compare in target_compare]
        print("Prediction completed. # test data: %d, "
              "total accuracy is %.1f%%, target accuracy is %.1f%%." % (len(y_test), accu, target_accu))
        print("The accuracy of the simple model:")
        print("\tKNeighborClassifier: total accuracy is %.1f%%, target accuracy is %.1f%%\n"
              "\tGradientBoostingClassifier: total accuracy is %.1f%%, target accuracy is %.1f%%\n"
              "\tRandomForestClassifier: total accuracy is %.1f%%, target accuracy is %.1f%%"
              % (accu_compare[0], accu_target_compare[0],
                 accu_compare[1], accu_target_compare[1],
                 accu_compare[2], accu_target_compare[2]))
        return y_pred


if __name__ == '__main__':
    X_cluster, X_classifier = get_data('大盘数据.xlsx')
    classifier = Classifier()
    X_test, y_test = classifier.fit(X_cluster, X_classifier)
    y_pred = classifier.predict(X_test, y_test)




