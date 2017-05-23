from sklearn.svm import SVC
import scipy.io as sio
import numpy as np

def mat_file_name(dataset, sigma, completion_alg):
    # gram_dataset_completionalg_sigma.mat
    return "gram_" + dataset + "_" + completion_alg + "_" + str(sigma) + ".mat"

def convert_index_to_attributes(index):
    type_, ground_truth, k_group, trial = index.split('_')
    return dict(type=type, ground_truth=ground_truth, k_group=k_group, trial=trial)

def separate_gram(gram, data_attributes, k_group):
    unmatched = []
    matched = []
    matched_is = []
    assert gram.__len__() == data_attributes.__len__()
    for i in range(gram.__len__()):
        if data_attributes[i].k_group == k_group:
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
   
   clf = SVC(C=cost, kernel='precomputed')
   clf.fit(np.array(train), np.array(train_gtruths))
   
   pred = clf.predict(validation_or_test)
   match = [z[0] == z[1] for z in zip(pred, v_or_t_gtruths)]
   return list(filter(lambda x: x is False, match)).__len__() / match.__len__()

def optimizehyperparameter(dataset,
                           completion_alg,
                           sigmas, # [sigma]
                           costs, # [C]
                           #train, # [person IDs], hence [k_group]
                           validation, # person ID/k_group
                           test # person ID/k_group
):
    error_to_hyperparameters = {}
    for sigma in sigmas:
        # sigma and completion_alg determines gram matrix file
        mat_file = mat_file_name(dataset, sigma, completion_alg) 
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
                                                        list(filter(lambda x: x.k_group != test, data_attributes)), validation)
        train_gtruths = list(filter(lambda attr: attr.k_group not in {validation, test}, data_attributes))
        validation_gtruths = list(filter(lambda attr: attr.k_group == validation, data_attributes))
        train_validation_gtruths = list(filter(lambda attr: attr.k_group != test, data_attributes))
        test_gtruths = list(filter(lambda attr: attr.k_group == test, data_attributes))
        
        for cost in costs:
            error_to_hyperparameters[tryout1hyperparameter(cost, train_matrix, train_gtruths,
                                                           validation_matrix, validation_gtruths)] = (sigma, cost)
            #/* indices in the gram matrix is passed to the function to indicate the split. */
        best_sigma, best_cost = error_to_hyperparameters[min(list(error_to_hyperparameters.keys()))]
    return tryout1hyperparameter(cost, train_validation_matrix, train_validation_gtruths, test_matrix, test_gtruths)

def crossvalidation(dataset, completion_alg, sigmas, costs):
    #for each split of gram into train/validation/test (for loop for 22 test subjects):
    # actually the dataset I have has 25 participants
    # ["A1", "C1", "C2", "C3", "C4", "E1", "G1", "G2", "G3", "I1", "I2", "I3",
    #  "J1", "J2", "J3", "L1", "M1", "S1", "T1", "U1", "Y1", "Y2", "Y3", "Z1", "Z2"]
    k_groups = ["A1", "C1", "C2", "C3", "C4", "E1", "G1", "G2", "G3", "I1", "I2", "I3",
                "J1", "J2", "J3", "L1", "M1", "S1", "T1", "U1", "Y1", "Y2", "Y3", "Z1", "Z2"]
    # actually participants/person ID.

    errors = []
    for i in range(k_groups.__len__()):
        validation_group = k_groups[i-1]
        test_group = k_groups[i]
        errors.append(optimizehyperparameter(dataset, completion_alg, sigmas, costs, validation_group, test_group))
        # /* indices in the gram matrix is passed to the function to indicate the split. */
    return np.average(errors)

def compare_completion_algorithms(dataset, sigmas, costs):
    result_no_completion = crossvalidation(dataset, "NO_COMPLETION", sigmas, costs)
    result_nuclear_norm_minimization = crossvalidation(dataset, "NUCLEAR_NORM_MINIMIZATION", sigmas, costs)
    result_soft_impute = crossvalidation(dataset, "SOFT_IMPUTE", sigmas, costs)

    print("NO_COMPLETION: " + repr(result_no_completion))
    print("NUCLEAR_NORM_MINIMIZATION: " + repr(result_nuclear_norm_minimization))
    print("SOFT_IMPUTE: " + repr(result_soft_impute))

if __name__ == "__main__":
    dataset = "upperChar"
    sigmas = [0.4, 1, 2, 5, 10]
    costs = []

    compare_completion_algorithms(dataset, sigmas, costs)

