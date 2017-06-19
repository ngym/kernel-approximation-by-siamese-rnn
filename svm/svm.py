import sys, json

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

def mat_file_name(sigma, completion_alg):
    # gram_dataset_completionalg_sigma.mat
    #print("loss:" + str(loss_persentage))
    if dataset_type == "upperChar":
        if completion_alg == "":
            return data_dir + \
                "gram_" + \
                dataset_type + "_" + \
                attribute_type + "_" + \
                "sigma" + ("%.3f" % sigma) + "_t1-t3" +\
                ".mat"
        #"sigma" + ("%.3f" % sigma) + \
        return data_dir + \
        "gram_" + \
        dataset_type + "_" + \
        attribute_type + "_" + \
        "sigma" + ("%.3f" % sigma) + "_t1-t3" + \
        "_loss" + str(loss_persentage) + "_" + \
        completion_alg + ".mat"
        #"sigma" + ("%.3f" % sigma) + \
    elif dataset_type == "UCIcharacter":
        if completion_alg == "":
            return data_dir + \
                "gram_" + \
                dataset_type + "_" + \
                "sigma" + ("%.3f" % sigma) + \
                ".mat"
        return data_dir + \
        "gram_" + \
        dataset_type + "_" + \
        "sigma" + ("%.3f" % sigma) + \
        "_loss" + str(loss_persentage) + "_" + \
        completion_alg + ".mat"
    elif dataset_type == "UCItctodd":
        if completion_alg == "":
            return data_dir + \
                "gram_" + \
                dataset_type + "_" + \
                "sigma" + ("%.3f" % sigma) + \
                ".mat"
        return data_dir + \
        "gram_" + \
        dataset_type + "_" + \
        "sigma" + ("%.3f" % sigma) + \
        "_loss" + str(loss_persentage) + "_" + \
        completion_alg + ".mat"
    else:
        assert False

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

def separate_gram(gram, data_attributes, k_group):
    unmatched = []
    matched = []
    matched_is = []
    assert gram.__len__() == data_attributes.__len__()
    for i in range(gram.__len__()):
        if data_attributes[i]['k_group'] == k_group:
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
    print(" " + functools.reduce(lambda a,b: a + "  " + b, [t for t in v_or_t_gtruths]))
    print(" " + functools.reduce(lambda a,b: a + "  " + b, [p for p in list(pred)]))
    print(" " + functools.reduce(lambda a,b: a + "  " + b,
                                 ["!" if z[0] != z[1] else " "
                                  for z in zip(list(pred), v_or_t_gtruths)]))
    print("---")
    return (roc_auc_score, f1_score)

def optimizehyperparameter(completion_alg,
                           sigmas, # [sigma]
                           costs, # [C]
                           #train, # [person IDs], hence [k_group]
                           validation, # person ID/k_group
                           test # person ID/k_group
):
    def train_validation_test_split(sigma):
        # sigma and completion_alg determines gram matrix file
        mat_file = mat_file_name(sigma, completion_alg) 
        mat = sio.loadmat(mat_file)
        gram = mat['gram']
        indices = mat['indices']

        data_attributes = []
        ground_truths = []
        for index in indices:
            attr = convert_index_to_attributes(index)
            data_attributes.append(attr)
            ground_truths.append(attr['ground_truth'])
            
        test_matrix, train_validation_matrix = separate_gram(gram, data_attributes, test)
        validation_matrix, train_matrix = separate_gram(train_validation_matrix,
                                                        [d for d in data_attributes if d['k_group'] != test],
                                                        validation)
        train_gtruths = [d['ground_truth'] for d in data_attributes if d['k_group'] not in {validation, test}]
        validation_gtruths = [d['ground_truth'] for d in data_attributes if d['k_group'] == validation]
        train_validation_gtruths = [d['ground_truth'] for d in data_attributes if d['k_group'] != test]
        test_gtruths = [d['ground_truth'] for d in data_attributes if d['k_group'] == test]
        return test_matrix, train_validation_matrix, validation_matrix, train_matrix,\
    train_gtruths, validation_gtruths, train_validation_gtruths, test_gtruths
        
    error_to_hyperparameters = {}
    for sigma in sigmas:
        test_matrix, train_validation_matrix, validation_matrix, train_matrix,\
    train_gtruths, validation_gtruths, train_validation_gtruths, test_gtruths\
    = train_validation_test_split(sigma)
        for cost in costs:
            error_to_hyperparameters[tryout1hyperparameter(cost, train_matrix, train_gtruths,
                                                           validation_matrix, validation_gtruths)] = (sigma, cost)
            #/* indices in the gram matrix is passed to the function to indicate the split. */
    best_sigma, best_cost = error_to_hyperparameters[max(list(error_to_hyperparameters.keys()))]
    test_matrix, train_validation_matrix, validation_matrix, train_matrix,\
    train_gtruths, validation_gtruths, train_validation_gtruths, test_gtruths\
    = train_validation_test_split(best_sigma)
    print("test")
    return tryout1hyperparameter(best_cost, train_validation_matrix, train_validation_gtruths, test_matrix, test_gtruths)

def crossvalidation(completion_alg, sigmas, costs):
    #for each split of gram into train/validation/test (for loop for 22 test subjects):
    # actually the dataset I have has 25 participants
    # ["A1", "C1", "C2", "C3", "C4", "E1", "G1", "G2", "G3", "I1", "I2", "I3",
    #  "J1", "J2", "J3", "L1", "M1", "S1", "T1", "U1", "Y1", "Y2", "Y3", "Z1", "Z2"]
    # actually participants/person ID.
    if dataset_type == "upperChar":
        k_groups = ["A1", "C1", "C2", "C3", "C4", "E1", "G1", "G2", "G3", "I1", "I2", "I3",
                    "J1", "J2", "J3", "L1", "M1", "S1", "T1", "U1", "Y1", "Y2", "Y3", "Z1", "Z2"]
        #k_groups = ["C1", "J1", "M1", "T1", "Y1", "Y2"]
    elif dataset_type == "UCItctodd":
        k_groups = [i for i in range(1, 10)]
    elif dataset_type == "UCIcharacter":
        k_groups = [i for i in range(10)]
    else:
        assert False
        
    errors = []
    for i in range(k_groups.__len__()):
        validation_group = k_groups[i-1]
        test_group = k_groups[i]
        errors.append(optimizehyperparameter(completion_alg, sigmas, costs, validation_group, test_group))
        # /* indices in the gram matrix is passed to the function to indicate the split. */
    return np.average([rocauc_f1[0] for rocauc_f1 in errors]), np.average([rocauc_f1[1] for rocauc_f1 in errors])

def compare_completion_algorithms(sigmas, costs):
    if loss_persentage == 0:
        result_ground_truth = crossvalidation("", sigmas, costs)
        print("Ground Truth: ROC_AUC:%.5f, F1:%.5f" % result_ground_truth)
    else:
        result_soft_impute = crossvalidation("SoftImpute", sigmas, costs)
        print("SoftImpute: ROC_AUC:%.5f, F1:%.5f" % result_soft_impute)
        #result_RNN_residual = crossvalidation("RNN_residual", sigmas, costs)
        #print("RNN_residual: ROC_AUC:%.5f, F1:%.5f" % result_RNN_residual)
        
        #result_nuclear_norm_minimization = crossvalidation("NuclearNormMinimization", sigmas, costs)
        #print("NuclearNormMinimization: " + repr(result_nuclear_norm_minimization))

if __name__ == "__main__":
    config_json_file = sys.argv[1]
    config_dict = json.load(open(config_json_file, 'r'))

    dataset_type = config_dict['dataset_type']
    attribute_type = config_dict['attribute_type']
    loss_persentage = config_dict['loss_persentage']
    sigmas_ = config_dict['gak_sigmas']
    l2regularization_costs = config_dict['l2regularization_costs']
    data_dir = config_dict['data_dir']
    
    compare_completion_algorithms(sigmas_, l2regularization_costs)
