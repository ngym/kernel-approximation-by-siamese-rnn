import functools
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, roc_auc_score

from datasets import others


def get_classification_error(dataset_type, gram, sample_names, test_indices, costs):
    train_and_validation_indices = [i for i in range(len(gram))
                                    if i not in test_indices]
    # TODO switch with read_sequences's list(key_to_str.values())
    ground_truth_labels = [others.get_label(dataset_type, sample_name) for sample_name in sample_names]
    train_and_validation_labels = [ground_truth_labels[i] for i in range(len(gram))
                                   if i not in test_indices]

    skf = StratifiedShuffleSplit(n_splits=1)
    tmp = skf.split(np.zeros_like(train_and_validation_labels), train_and_validation_labels)
    validation_indices = []
    for t in next(tmp)[1]:
        validation_indices.append(train_and_validation_indices[t])

    return optimize_hyperparameter(costs,
                                  gram,
                                  ground_truth_labels,
                                  validation_indices,
                                  test_indices)


def optimize_hyperparameter(costs,
                            gram,
                            labels,
                            validation_indices,
                            test_indices):
    error_to_hyperparameters = {}
    (train_matrix, train_labels), (validation_matrix, validation_labels), \
        (train_and_validation_matrix, train_and_validation_labels), (test_matrix, test_labels) \
        = train_validation_test_split(gram, labels, validation_indices, test_indices)
    for cost in costs:
        error_to_hyperparameters[tryout_hyperparameter(cost, train_matrix,
                                                       train_labels,
                                                       validation_matrix,
                                                       validation_labels)] = (cost)
    #/* indices in the gram matrix is passed to the function to indicate the split. */
    best_cost = error_to_hyperparameters[max(list(error_to_hyperparameters.keys()))]
    print("test")
    return tryout_hyperparameter(best_cost, train_and_validation_matrix,
                                 train_and_validation_labels, test_matrix, test_labels)


def train_validation_test_split(gram, labels, validation_indices, test_indices):
    length = len(gram)
    train_and_validation_matrix = [[gram[i][j]
                                for j in range(length) if j not in test_indices]
                               for i in range(length) if i not in test_indices]
    test_matrix = [[gram[i][j]
                    for j in range(length) if j not in test_indices]
                   for i in range(length) if i in test_indices]

    train_matrix = [[gram[i][j]
                     for j in range(length) if j not in np.r_[validation_indices, test_indices]]
                    for i in range(length) if i not in np.r_[validation_indices, test_indices]]
    validation_matrix = [[gram[i][j]
                          for j in range(length) if j not in np.r_[validation_indices,
                                                                   test_indices]]
                         for i in range(length) if i in validation_indices]

    train_labels = [labels[i] for i in range(len(labels))
                    if i not in validation_indices and i not in test_indices]
    validation_labels = [labels[i] for i in range(len(labels))
                         if i in validation_indices]
    train_and_validation_labels = [labels[i] for i in range(len(labels))
                                   if i not in test_indices]
    test_labels = [labels[i] for i in range(len(labels))
                   if i in test_indices]

    print(np.array(train_and_validation_matrix).shape)
    print(np.array(test_matrix).shape)
    print(np.array(train_matrix).shape)
    print(np.array(validation_matrix).shape)

    print(np.array(train_and_validation_labels).shape)
    print(np.array(test_labels).shape)
    print(np.array(train_labels).shape)
    print(np.array(validation_labels).shape)

    return (train_matrix, train_labels), (validation_matrix, validation_labels), \
           (train_and_validation_matrix, train_and_validation_labels), (test_matrix, test_labels)


def tryout_hyperparameter(cost, train, train_labels, validation_or_test, validation_or_test_labels):
    # indices in the gram matrix is passed to the function to indicate the split.
    clf = SVC(C=cost, kernel='precomputed', probability=True)
    clf.fit(np.array(train), np.array(train_labels))

    predicted_labels = clf.predict(validation_or_test)
    f1_ = f1_score(validation_or_test_labels, predicted_labels, average='weighted')

    lb = LabelBinarizer()
    lb.fit(train_labels)
    y_true = lb.transform(validation_or_test_labels)
    predicted_probabilities = clf.predict_proba(validation_or_test)
    auc_ = roc_auc_score(y_true=y_true, y_score=predicted_probabilities)

    print("l2regularization_costs: " + repr(cost))
    print("f1_score: " + repr(f1_))
    print("roc_auc_score:" + repr(auc_))
    print(" " + functools.reduce(lambda a, b: a + "  " + b, [t for t in validation_or_test_labels]))
    print(" " + functools.reduce(lambda a, b: a + "  " + b, [p for p in list(predicted_labels)]))
    print(" " + functools.reduce(lambda a, b: a + "  " + b,
                                 ["!" if z[0] != z[1] else " "
                                  for z in zip(list(predicted_labels), validation_or_test_labels)]))
    print("---")

    return (auc_, f1_)