# -*- coding: UTF-8 -*-

import numpy as np
import xlrd
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score


def get_data(filename):
    wb = xlrd.open_workbook(filename)
    ws = wb.sheet_by_name('Sheet1')
    l = len(ws.col_values(2)[2:])
    cols = [scale(ws.col_values(i)[2:]) for i in range(7, 37) if i != 29]
    col1, col2 = ws.col_values(2)[2:], ws.col_values(5)[2:]
    X_cluster = [[col1[i], col2[i]] for i in range(l) if col2[i] < 20]
    X_classifier = np.array([[cols[j][i] for j in range(len(cols))] for i in range(l) if col2[i] < 20])
    print("Loading of datas completed. # data: %d" % len(X_classifier))
    return X_cluster, X_classifier


def divide_data(X, y, ratio):
    size = int(ratio * len(X))
    return X[:size], X[size:], y[:size], y[size:]


def choose_label(X, y, label):
    X_, y_ = [], []
    for i in range(len(X)):
        if y[i] == label:
            X_.append(X[i])
            y_.append(y[i])
    X_, y_ = np.array(X_), np.array(y_)
    return X_, y_


def separate(X, y):
    labels = set(y)
    classes = {label: choose_label(X, y, label) for label in labels}
    datas = {}
    for label1 in labels:
        X1, y1 = classes[label1]
        for label2 in labels:
            if label1 < label2:
                X2, y2 = classes[label2]
                X_, y_ = np.concatenate((X1, X2), axis=0), np.concatenate((y1, y2), axis=0)
                datas[(label1, label2)] = (X_, y_)
    return datas


def max_count(array):
    return max(array, key=array.count)


def get_target_labels(X, y, n_label):
    datas = [choose_label(X, y, i) for i in range(n_label)]
    X_ = [(d[1][0], [x[1] for x in d[0]]) for d in datas]
    averages = [(np.mean(x[1]), x[0]) for x in X_]
    target_labels = [min(averages, key=lambda x: x[0])[1], max(averages, key=lambda x: x[0])[1]]
    print(target_labels)
    target_labels.sort()
    return target_labels


if __name__ == "__main__":
    # 可视化数据

    X_cluster, X_classifier = get_data('大盘数据.xlsx')
    X0, X1 = [], []
    for i in range(len(X_cluster)):
        X0.append(X_cluster[i][0])
        X1.append(X_cluster[i][1])
    plt.figure()
    plt.scatter(X0[:], X1[:])
    plt.xlabel('Change of Index')
    plt.ylabel('Ratio of Stocks Rising and Falling')

    # 最优聚类方式
    plt.figure(figsize=(8, 2))
    algorithms = [
        ('K = ' + str(n_clusters), SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', gamma=10))
        for n_clusters in range(2, 6)]
    plot_num = 1
    for name, algorithm in algorithms:
        algorithm.fit(X_cluster)
        y_pred = algorithm.labels_.astype(np.int)
        score = silhouette_score(X_cluster, y_pred)
        plt.subplot(1, len(algorithms), plot_num)
        plt.title(name)
        plt.scatter(X0[:], X1[:], c=y_pred)
        plt.text(.99, .01, ('score: %.2f' % score).lstrip('0'), transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1
    plt.show()

