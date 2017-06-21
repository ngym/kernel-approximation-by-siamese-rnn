import sys, json, glob, os

from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
import sklearn.metrics as metrics
import scipy.io as sio
import numpy as np
import functools

dataset_type = None
attribute_type = None
loss_persentage = None
data_dir = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/program/gak_gram_matrix_completion/OUTPUT/"

def convert_index_to_attributes(index):
    if dataset_type == "upperChar":
        # 6DMG
        index_ = index.split('/')[-1]
        type_, ground_truth, k_group, trial = index_.split('_')
    elif dataset_type == "UCItctodd":
        # UCI AUSLAN
        index_ = index.split('/')[-1]
        k_group = int(index_.split('-')[-2])
        ground_truth = functools.reduce(lambda a, b: a + "-" + b, index_.split('-')[:-2])
        type_ = None
        trial = None
    elif dataset_type == "UCIcharacter":
        # UCI character
        ground_truth = index[0]
        serial_number = int(index[1:])
        k_group = serial_number % 10
        type_ = None
        trial = None
    else:
        assert False
    return dict(type=type_, ground_truth=ground_truth, k_group=k_group, trial=trial)

def separate_gram(gram, indices_to_separate):
    unmatched = []
    matched = []
    matched_is = []
    for i in range(gram.__len__()):
        if i in indices_to_separate:
            matched.append(gram[i])
            matched_is.append(i)
        else:
            unmatched.append(gram[i])
    new_matched = []
    for row in matched:
        new_row_matched = []
        for i in range(gram.__len__()):
            if i not in matched_is:
                new_row_matched.append(row[i])
        new_matched.append(new_row_matched)
    new_unmatched = []
    for row in unmatched:
        new_row_unmatched = []
        for i in range(gram.__len__()):
            if i not in matched_is:
                new_row_unmatched.append(row[i])
        new_unmatched.append(new_row_unmatched)
    return new_matched, new_unmatched

def train_validation_test_split(gtruths, validation_indices, test_indices):
    test_matrix, train_validation_matrix = separate_gram(gram, test_indices)
    validation_matrix, train_matrix = separate_gram(train_validation_matrix,
                                                    validation_indices)

    train_gtruths = [gtruths[i] for i in range(len(gtruths))
                     if i not in validation_indices and i not in test_indices]
    validation_gtruths = [gtruths[i] for i in range(len(gtruths))
                          if i in validation_indices]
    train_validation_gtruths = [gtruths[i] for i in range(len(gtruths))
                                if i not in test_indices]
    test_gtruths = [gtruths[i] for i in range(len(gtruths))
                    if i in test_indices]
    return test_matrix, train_validation_matrix, validation_matrix, train_matrix,\
        train_gtruths, validation_gtruths, train_validation_gtruths, test_gtruths

def tryout1hyperparameter(cost, train, train_gtruths, validation_or_test, v_or_t_gtruths):
    # indices in the gram matrix is passed to the function to indicate the split.
    clf = SVC(C=cost, kernel='precomputed', probability=True)
    clf.fit(np.array(train), np.array(train_gtruths))
    pred = clf.predict(validation_or_test)
    f1_score = metrics.f1_score(v_or_t_gtruths, pred, average='weighted')
    pred_prob = clf.predict_proba(validation_or_test)
    lb = LabelBinarizer()
    y_true = lb.fit_transform(v_or_t_gtruths)
    assert all(lb.classes_ == clf.classes_)
    roc_auc_score = metrics.roc_auc_score(y_true=y_true, y_score=pred_prob)
    print("l2regularization_costs: " + repr(cost))
    print("f1_score: " + repr(f1_score))
    print("roc_auc_score:" + repr(roc_auc_score))
    #print([int(n) for n in list(pred)])
    #print([int(n) for n in v_or_t_gtruths])
    print(" " + functools.reduce(lambda a, b: a + "  " + b, [t for t in v_or_t_gtruths]))
    print(" " + functools.reduce(lambda a, b: a + "  " + b, [p for p in list(pred)]))
    print(" " + functools.reduce(lambda a, b: a + "  " + b,
                                 ["!" if z[0] != z[1] else " "
                                  for z in zip(list(pred), v_or_t_gtruths)]))
    print("---")
    return (roc_auc_score, f1_score)

def optimizehyperparameter(costs, # [C]
                           validation_indices,
                           test_indices):
    error_to_hyperparameters = {}
    test_matrix, train_validation_matrix, validation_matrix, train_matrix,\
        train_gtruths, validation_gtruths, train_validation_gtruths, test_gtruths\
        = train_validation_test_split(gtruths, validation_indices, test_indices)
    for cost in costs:
        error_to_hyperparameters[tryout1hyperparameter(cost, train_matrix,
                                                       train_gtruths,
                                                       validation_matrix,
                                                       validation_gtruths)] = (sigma,
                                                                               cost)
    #/* indices in the gram matrix is passed to the function to indicate the split. */
    best_sigma, best_cost = error_to_hyperparameters[max(list(error_to_hyperparameters.keys()))]
    test_matrix, train_validation_matrix, validation_matrix, train_matrix,\
        train_gtruths, validation_gtruths, train_validation_gtruths, test_gtruths\
        = train_validation_test_split(gtruths, validation_indices, test_indices)
    print("test")
    return tryout1hyperparameter(best_cost, train_validation_matrix,
                                 train_validation_gtruths, test_matrix, test_gtruths)

def crossvalidation(mat_file_names, costs):
    errors = []
    for mat_file_name in mat_file_names:
        mat = io.loadmat(mat_file_name)
        gram = mat['gram']
        num_indices = len(gram)

        test_indices = mat['dropped_indices_number']
        validation_indices = []

        for i in range(num_indices):
            if i not in test_indices:
                validation_indices.append(i)

        errors.append(optimizehyperparameter(costs,
                                             validation_indices,
                                             test_indices))

    print("ROC_AUCs:")
    print([rocauc_f1[0] for rocauc_f1 in errors])
    print("F1s:")
    print([rocauc_f1[1] for rocauc_f1 in errors])
    scores = np.average([rocauc_f1[0] for rocauc_f1 in errors]),\
             np.average([rocauc_f1[1] for rocauc_f1 in errors])
    print("Average ROC_AUC:%.5f, Average F1:%.5f" % scores)

def main():
    config_json_file = sys.argv[1]
    config_dict = json.load(open(config_json_file, 'r'))

    data_dir = config_dict['data_dir']
    l2regularization_costs = config_dict['l2regularization_costs']

    mat_file_names = []
    for file_name_for_glob in config_dict['completed_matrices_for_glob']:
        for mat_file_name in glob.glob(os.path.join(data_dir,
                                                    file_name_for_glob)):
            mat_file_names.append(mat_file_name)
            print(mat_file_name)

    crossvalidation(mat_file_names, l2regularization_costs)

if __name__ == "__main__":
    main()
