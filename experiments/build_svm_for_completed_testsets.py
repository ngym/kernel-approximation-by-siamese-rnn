import sys, json, os, subprocess, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def main():
    PROGRAM = "/home/milacski/shota/fast-time-series-data-classification/algorithms/svm_for_completed_testsets.py"
    l2regularization_costs = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64],
    
    experiments_dir = "/home/milacski/shota/USE_CASE_RNN_COMPLETION_1_VALIDATION",
    for dataset_type, gram_file in [("6DMG", "gram_upperChar_all_sigma20.000_t1-t3_dropfrom0_RNN_LSTM.pkl"),
                         ("UCIauslan", "gram_UCIauslan_sigma12.000_dropfrom0_RNN_LSTM.pkl"),
                         ("UCIcharacter", "gram_UCIcharacter_sigma20.000_dropfrom0_RNN_LSTM.pkl")]:
        data_dir = os.path.join(experiments_dir, dataset_type,
                                "/LSTM/Forward/BatchNormalization/[10]/[3]/0.3/0")
        output_file = gram_file.replace('.pkl', '_svm_out.json')
        json_dict = dict(dataset_type=dataset_type,
                         data_dir=data_dir,
                         completion_matrices_for_glob=[gram_file],
                         l2regularization_costs=l2regularization_costs,
                         output_file=output_file)
    
        json_file_name = os.path.join(data_dir, "svm_for_completed_testsets.json")
        fd = open(json_file_name, "w")
        json.dump(json_dict, fd)
        fd.close()
    
    experiment_dirs = ["/home/milacski/shota/USE_CASE_GAK_COMPLETION_1_VALIDATION",
                       "/home/milacski/shota/USE_CASE_SOFTIMPUTE_COMPLETION_1_VALIDATION"]
    for experiment_dir in experiment_dirs:
        for dataset_type, direc, gram_file in [("6DMG", "6DMGupperChar/all/20/0", "gram_upperChar_all_sigma20.000_t1-t3_dropfrom0_RNN_LSTM.pkl"),
                                               ("UCIauslan", "UCIauslan/12/0", "gram_UCIauslan_sigma12.000_dropfrom0_RNN_LSTM.pkl"),
                                               ("UCIcharacter", "UCIcharacter/20/0", "gram_UCIcharacter_sigma20.000_dropfrom0_RNN_LSTM.pkl")]:
            data_dir = os.path.join(experiments_dir, direc)
            output_file = gram_file.replace('.pkl', '_svm_out.json')
            json_dict = dict(dataset_type=dataset_type,
                             data_dir=data_dir,
                             completion_matrices_for_glob=[gram_file],
                             l2regularization_costs=l2regularization_costs,
                             output_file=output_file)
    
            json_file_name = os.path.join(data_dir, "svm_for_completed_testsets.json")
            fd = open(json_file_name, "w")
            json.dump(json_dict, fd)
            fd.close()
    

if '__name__' == '__main__':
    main()

    
