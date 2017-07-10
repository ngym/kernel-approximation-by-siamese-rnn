import sys, os
import glob

from sacred import Experiment

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from algorithms import svm
from utils import file_utils

ex = Experiment('compute_classification_errors')


@ex.config
def cfg():
    pickle_files = "results/*.pkl"
    regularization_costs = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    output_file = "results/classification_costs.json"


@ex.automain
def run(pickle_files, regularization_costs, output_file):
    output_file = os.path.abspath(output_file)
    pickle_files = glob.glob(pickle_files)

    dic = {}
    for pickle_file in pickle_files:
        pkl = file_utils.load_pickle(os.path.abspath(pickle_file))

        dataset_type = pkl['dataset_type']
        gram_matrices = pkl['gram_matrices']
        if len(gram_matrices) == 1:
            continue
        gram = gram_matrices[-1]['completed_npsd']
        sample_names = pkl['sample_names']
        test_indices = pkl['dropped_indices'][-1]

        (roc_auc_score, f1_score) = svm.get_classification_error(dataset_type,
                                                             gram,
                                                             sample_names,
                                                             test_indices,
                                                             regularization_costs)
        print(pickle_file + " roc_auc_score: " + str(roc_auc_score) + " f1_score: " + str(f1_score))
        dic[pickle_file] = dict(roc_auc_score=roc_auc_score,
                                f1_score=f1_score)

    file_utils.save_json(output_file, dic)