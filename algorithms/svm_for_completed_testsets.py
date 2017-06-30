import sys, json, glob, os
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, roc_auc_score
import scipy.io as io
import numpy as np
import functools
import pickle

dataset_type = None
attribute_type = None

def convert_index_to_attributes(index):
    if dataset_type in {"upperChar", "6DMGupperChar", "6DMG"}:
        # 6DMG
        index_ = index.split('/')[-1]
        type_, ground_truth, k_group, trial = index_.split('_')
    elif dataset_type in {"UCItctodd", "UCIauslan"}:
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

def train_validation_test_split(gram, gtruths, validation_indices, test_indices):
    length = len(gram)
    train_validation_matrix = [[gram[i][j]
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

    train_gtruths = [gtruths[i] for i in range(len(gtruths))
                     if i not in validation_indices and i not in test_indices]
    validation_gtruths = [gtruths[i] for i in range(len(gtruths))
                          if i in validation_indices]
    train_validation_gtruths = [gtruths[i] for i in range(len(gtruths))
                                if i not in test_indices]
    test_gtruths = [gtruths[i] for i in range(len(gtruths))
                    if i in test_indices]

    print(np.array(train_validation_matrix).shape)
    print(np.array(test_matrix).shape)
    print(np.array(train_matrix).shape)
    print(np.array(validation_matrix).shape)
    
    print(np.array(train_validation_gtruths).shape)
    print(np.array(test_gtruths).shape)
    print(np.array(train_gtruths).shape)
    print(np.array(validation_gtruths).shape)
    
    return test_matrix, train_validation_matrix, validation_matrix, train_matrix,\
        train_gtruths, validation_gtruths, train_validation_gtruths, test_gtruths

def tryout1hyperparameter(cost, train, train_gtruths, validation_or_test, v_or_t_gtruths):
    # indices in the gram matrix is passed to the function to indicate the split.
    
    
    clf = SVC(C=cost, kernel='precomputed', probability=True)
    clf.fit(np.array(train), np.array(train_gtruths))
    pred = clf.predict(validation_or_test)
    f1_ = f1_score(v_or_t_gtruths, pred, average='weighted')
    pred_prob = clf.predict_proba(validation_or_test)
    lb = LabelBinarizer()
    lb.fit(train_gtruths)
    y_true = lb.transform(v_or_t_gtruths)
    #lb = LabelBinarizer()
    #y_true = lb.fit_transform(v_or_t_gtruths)
    assert all(lb.classes_ == clf.classes_)
    auc_ = roc_auc_score(y_true=y_true, y_score=pred_prob)
    print("l2regularization_costs: " + repr(cost))
    print("f1_score: " + repr(f1_))
    print("roc_auc_score:" + repr(auc_))
    #print([int(n) for n in list(pred)])
    #print([int(n) for n in v_or_t_gtruths])
    print(" " + functools.reduce(lambda a, b: a + "  " + b, [t for t in v_or_t_gtruths]))
    print(" " + functools.reduce(lambda a, b: a + "  " + b, [p for p in list(pred)]))
    print(" " + functools.reduce(lambda a, b: a + "  " + b,
                                 ["!" if z[0] != z[1] else " "
                                  for z in zip(list(pred), v_or_t_gtruths)]))
    print("---")
    return (auc_, f1_)

def optimizehyperparameter(costs, # [C]
                           gram,
                           gtruths,
                           validation_indices,
                           test_indices):
    error_to_hyperparameters = {}
    test_matrix, train_validation_matrix, validation_matrix, train_matrix,\
        train_gtruths, validation_gtruths, train_validation_gtruths, test_gtruths\
        = train_validation_test_split(gram, gtruths, validation_indices, test_indices)
    for cost in costs:
        error_to_hyperparameters[tryout1hyperparameter(cost, train_matrix,
                                                       train_gtruths,
                                                       validation_matrix,
                                                       validation_gtruths)] = (cost)
    #/* indices in the gram matrix is passed to the function to indicate the split. */
    best_cost = error_to_hyperparameters[max(list(error_to_hyperparameters.keys()))]
    test_matrix, train_validation_matrix, validation_matrix, train_matrix,\
        train_gtruths, validation_gtruths, train_validation_gtruths, test_gtruths\
        = train_validation_test_split(gram, gtruths, validation_indices, test_indices)
    print("test")
    return tryout1hyperparameter(best_cost, train_validation_matrix,
                                 train_validation_gtruths, test_matrix, test_gtruths)

def crossvalidation(pkl_file_names, costs):
    errors = []
    for pkl_file_name in pkl_file_names:
        fd = open(pkl_file_name, 'rb')
        pkl = pickle.load(fd)
        gram = pkl['gram_matrices'][-1]['gram_completed_npsd']
        num_indices = len(gram)
        indices = pkl['sample_names']

        test_indices = pkl['drop_indices']
        tr_and_v_indices = []

        for i in range(num_indices):
            if i not in test_indices:
                tr_and_v_indices.append(i)
                
        gtruths = [convert_index_to_attributes(index)['ground_truth'] for index in indices]
        train_validation_gtruths = [gtruths[i] for i in range(len(gtruths))
                                    if i not in test_indices]
        
        #validation_indices = np.random.permutation(tr_and_v_indices)\
        #                     [:(len(tr_and_v_indices)//9)]
        skf = StratifiedShuffleSplit(n_splits=1)
        tmp = skf.split(np.zeros_like(train_validation_gtruths),
                        train_validation_gtruths)
        validation_indices = []
        for t in next(tmp)[1]:
            validation_indices.append(tr_and_v_indices[t])

        errors.append(optimizehyperparameter(costs,
                                             gram,
                                             gtruths,
                                             validation_indices,
                                             test_indices))
    print("ROC_AUCs:")
    print([rocauc_f1[0] for rocauc_f1 in errors])
    print("F1s:")
    print([rocauc_f1[1] for rocauc_f1 in errors])
    ave_roc = np.average([rocauc_f1[0] for rocauc_f1 in errors])
    ave_f1 = np.average([rocauc_f1[1] for rocauc_f1 in errors])
    print("Average ROC_AUC:%.5f, Average F1:%.5f" % (ave_roc, ave_f1))

    return ave_roc, ave_f1

def main():
    config_json_file = sys.argv[1]
    config_dict = json.load(open(config_json_file, 'r'))
    global dataset_type
    dataset_type = config_dict['dataset_type']

    data_dir = config_dict['data_dir']
    l2regularization_costs = config_dict['l2regularization_costs']
    output_file = config_dict['output_file']

    pkl_file_names = []
    for file_name_for_glob in config_dict['completed_matrices_for_glob']:
        for pkl_file_name in glob.glob(os.path.join(data_dir,
                                                    file_name_for_glob)):
            pkl_file_names.append(pkl_file_name)
            print(pkl_file_name)

    ave_roc, ave_f1 = crossvalidation(pkl_file_names, l2regularization_costs)

    json_dict = {}
    json_dict['ROC_AUC'] = ave_roc
    json_dict['F1'] = ave_f1
    fd = open(output_file, "w")
    json.dump(json_dict, fd)
    fd.close()

if __name__ == "__main__":
    main()
