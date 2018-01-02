import sys, json, os
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer 
from sklearn.metrics import f1_score, roc_auc_score
import scipy.io as io
import numpy as np
import functools
import pickle

def compute_classification_errors(train_validation_features,
                                  train_validation_labels,
                                  test_features,
                                  test_labels):
    regularization_costs = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]

    best_cost = select_good_cost(regularization_costs,
                            train_validation_features,
                            train_validation_labels)
    
    auc, f1, time_classification_start, time_classification_end = linear_svm(best_cost, train_validation_features,
                                                                         train_validation_labels,
                                                                         test_features,
                                                                         test_labels)
    print("best_cost:%f" % best_cost)
    return auc, f1, time_classification_start, time_classification_end

def select_good_cost(regularization_costs,
                     train_validation_features, train_validation_labels):
    skf = StratifiedShuffleSplit(n_splits=1)
    tmp = skf.split(np.zeros_like(train_validation_labels), train_validation_labels)
    validation_indices = []
    for t in next(tmp)[1]:
        validation_indices.append(t)

    train_indices = np.delete(np.arange(len(train_validation_features)), validation_indices)

    train_features = train_validation_features[train_indices]
    validation_features = train_validation_features[validation_indices]

    train_labels = train_validation_labels[train_indices]
    validation_labels = train_validation_labels[validation_indices]

    best_auc = 0
    best_f1 = 0
    best_cost = regularization_costs[0]
    for cost in regularization_costs:
        auc, f1, _, _ = linear_svm(cost,
                             train_features, train_labels,
                             validation_features, validation_labels)
        if auc > best_auc or (auc == best_auc and f1 > best_f1):
            best_auc = auc
            best_f1 = f1
            best_cost = cost
    return cost
    
def linear_svm(cost, train_features, train_labels,
               validation_test_features,
               validation_test_labels):
    #clf = LinearSVC()
    clf = SVC(C=cost, kernel='linear', probability=True)

    clf.fit(train_features, train_labels)

    time_classification_start = os.times()
    predicted_labels = clf.predict_on_batch(validation_test_features)
    time_classification_end = os.times()

    
    f1 = f1_score(validation_test_labels, predicted_labels, average='weighted')
    
    lb = LabelBinarizer()
    lb.fit(train_labels)
    y_true = lb.transform(validation_test_labels)
    predicted_probabilities = clf.predict_proba(validation_test_features)
    auc = roc_auc_score(y_true=y_true, y_score=predicted_probabilities)

    return auc, f1, time_classification_start, time_classification_end





