import sys, os
import glob

from sacred import Experiment

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from algorithms import svm
from utils import file_utils

ex = Experiment('compute_classification_errors')


@ex.config
def cfg():
    pickle_or_hdf5_locations = "results/*.pkl"
    regularization_costs = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    output_file = "results/classification_costs.json"


@ex.automain
def run(pickle_or_hdf5_locations, regularization_costs, output_file):
    output_file = os.path.abspath(output_file)
    pickle_or_hdf5_locations = glob.glob(pickle_or_hdf5_locations)

    dic = {}
    for pickle_or_hdf5_location in pickle_or_hdf5_locations:
        hdf5 = pickle_or_hdf5_location[-4:] == "hdf5"
        if hdf5:
            loaded_data = file_utils.load_hdf5(os.path.abspath(pickle_or_hdf5_location))
        else:
            loaded_data = file_utils.load_pickle(os.path.abspath(pickle_or_hdf5_location))

        dataset_type = loaded_data['dataset_type']
        gram_matrices = loaded_data['gram_matrices']
        if len(gram_matrices) == 1:
            continue
        gram = gram_matrices[-1]['completed_npsd']
        sample_names = loaded_data['sample_names']
        test_indices = loaded_data['dropped_indices'][-1]

        (roc_auc_score, f1_score) = svm.get_classification_error(dataset_type,
                                                             gram,
                                                             sample_names,
                                                             test_indices,
                                                             regularization_costs)
        print(pickle_or_hdf5_location + " roc_auc_score: " + str(roc_auc_score) + " f1_score: " + str(f1_score))
        dic[pickle_or_hdf5_location] = dict(roc_auc_score=roc_auc_score,
                                            f1_score=f1_score)

    file_utils.save_json(output_file, dic)


    
