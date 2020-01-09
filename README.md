# FML_Project
This is the FINAL project of the Fundamental Machine Learning course in 2019.

## Introduction
The codes are divided into two parts, i.e., data processing and vote classification. By this project, you can

- visualize the data and the cluster results by running data_processing.py
- get the prediction result of the vote model and the final accuracy by running vote_classifier.py

## Usage

### data_processing.py

Simply run this code you will get the figure of the data and the cluster results with different hyperparameters `n_clusters`

### vote_classifier.py

This file is dependent on data_processing.py, please be sure these two files are in the same folder.

There are two classes in this file, namely `VoteClassifier` and `Classifier`. The former is the weighted voting binary classifier while the latter consists of distinct `VoteClassifier`'s for each group.

#### `VoteClassifier`

The class is based on three base classifier listed below:

- `KNeighbosClassifier`
- `GradientBoostingClassifier`
- `RandomForestClassifier`

To initialize it, simply use its constructor together with some hyperparameters of the three base classifier. You can set the most important hyperparameters of the three base models.

```
__init__(self, knn_n_neighbors=5, gb_n_estimators=400, rf_n_estimators='warn', rf_oob_score=False)
```

To fit and predict, simply use `fit(self, X, y)` and `predict(self, X)` similar to any other models, it will automatically calculate the weights and do the linear combination.

#### `Classifier`

The class consists of distinct `VoteClassifier`'s for each group, you can set the hyperparameters `n_label` when initializing which we do not recommand you to do so. We here simply use the default constructor to initialize each `VoteClassifier`.

Since we here need to cluster the data during fitting the model, we also put the data splitting into the fitting process, so the input of `fit(self, X_cluster, X_classifier, ratio=.2)` will not only contain the data for training, but also the data for clustering and the split ratio. It will return the test data for validation.

Similarly, we need to calculate all related accuracy so we put these calculation into the predicting process in order to avoid some troubles, and the input of `predict(self, X_test, y_test)` will also contain the correct labels. However, it will still output the prediction result, as all the models will do.
