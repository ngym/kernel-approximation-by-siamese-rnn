import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time, pickle, json
import numpy as np
from scipy import io
from fancyimpute import SoftImpute

from utils.nearest_positive_semidefinite import nearest_positive_semidefinite
from utils.errors import mean_squared_error
from utils.plot_gram_to_html import plot_gram_to_html
from utils.make_matrix_incomplete import gram_drop_random

from utils.nearest_positive_semidefinite import nearest_positive_semidefinite
from utils.errors import mean_squared_error, mean_absolute_error, relative_error
from utils.plot_gram_to_html import plot_gram_to_html
from utils.make_matrix_incomplete import gram_drop_random, gram_drop_samples
from datasets.read_sequences import read_sequences
from utils.plot_gram_to_html import plot_gram_to_html

def softimpute_matrix_completion(gram_drop):
    """Fill in Gram matrix with dropped elements with Soft Impute Matrix Completion.
    Optimizes the Matrix Completion objective using Singular Value Thresholding

    :param gram_drop: Gram matrix with dropped elements
    :type gram_drop: list of lists
    :returns: Filled in Gram matrix, optimization start and end times
    :rtype: list of lists, float, float, float, float
    """
    t_start = time.time()
    gram_completed = SoftImpute().complete(gram_drop)
    t_end = time.time()
    return gram_completed, t_start, t_end
    
def main():
    main_start = time.time()
    if len(sys.argv) != 2:
        random_drop = True
        gram_filename = sys.argv[1]
        drop_percent = int(sys.argv[2])
        completionanalysisfile = sys.argv[3]
    else:
        random_drop = False
        config_json_file = sys.argv[1]
        config_dict = json.load(open(config_json_file, 'r'))

        gram_filename = config_dict['gram_file']
        indices_to_drop = config_dict['indices_to_drop']
        completionanalysisfile = config_dict['completionanalysisfile']

    fd = open(gram_filename, 'rb')
    pkl = pickle.load(fd)
    fd.close()
    
    dataset_type = pkl['dataset_type']
    gram_matrices = pkl['gram_matrices']
    if len(gram_matrices) == 1:
        gram = gram_matrices[0]['gram_original']
    else:
        gram = gram_matrices[-1]['gram_completed_npsd']
        
    sample_names = pkl['sample_names']

    logfile_loss = completionanalysisfile.replace(".timelog", ".losses")

    seed = 1

    fd = open(completionanalysisfile, "w")

    if random_drop:
        gram_drop, dropped_elements = gram_drop_random(seed, gram, drop_percent)
        logfile_html = gram_filename.replace(".pkl", "_drop" + str(drop_percent) + "_SoftImpute_Completion.html")
        logfile_pkl  = gram_filename.replace(".pkl", "_drop" + str(drop_percent) + "_SoftImpute_Completion.pkl")
    else:
        gram_drop, dropped_elements = gram_drop_samples(gram, indices_to_drop)
        logfile_html = gram_filename.replace(".pkl", "_dropfrom" + str(indices_to_drop[0]) + "_SoftImpute_Completion.html")
        logfile_pkl  = gram_filename.replace(".pkl", "_dropfrom" + str(indices_to_drop[0]) + "_SoftImpute_Completion.pkl")

    fd.write("number of dropped elements: " + str(len(dropped_elements)))
    fd.write("\n")

    # Soft Impute
    gram_completed, softimpute_start, softimpute_end = softimpute_matrix_completion(gram_drop)
    # eigenvalue check
    npsd_start = time.time()
    gram_completed_npsd = nearest_positive_semidefinite(gram_completed)
    npsd_end = time.time()

    # OUTPUT
    plot_gram_to_html(logfile_html,
                      gram_completed_npsd, sample_names)

    new_gram_matrices = {"gram_completed_npsd": np.array(gram_completed_npsd),
                         "gram_completed": np.array(gram_completed),
                         "gram_drop": np.array(gram_drop)}
    gram_matrices.append(new_gram_matrices)
    mat_log = pkl['log']
    new_log = "command: " + "".join(sys.argv) + time.asctime(time.localtime())
    mat_log.append(new_log)

    drop_indices = pkl['drop_indices']
    drop_indices.append(dropped_elements)

    pkl_fd = open(logfile_pkl, 'wb')
    dic = dict(gram_matrices=gram_matrices,
               drop_indices=drop_indices,
               dataset_type=dataset_type,
               log=mat_log,
               sample_names=sample_names)
    pickle.dump(dic, pkl_fd)
    pkl_fd.close()

    print("SoftImpute is output")

    mse = mean_squared_error(gram, gram_completed_npsd)
    msede = mean_squared_error(gram,
                               gram_completed_npsd,
                               dropped_elements)

    mae = mean_absolute_error(gram, gram_completed_npsd)
    maede = mean_absolute_error(gram,
                                gram_completed_npsd,
                                dropped_elements)
    
    re = relative_error(gram,
                        gram_completed_npsd)
    rede = relative_error(gram,
                         gram_completed_npsd,
                         dropped_elements)

    main_end = time.time()

    analysis_json = {}
    analysis_json['number_of_dropped_elements'] = len(dropped_elements)
    num_calculated_elements = len(dropped_elements) - len(indices_to_drop) // 2
    analysis_json['number_of_calculated_elements'] = num_calculated_elements
    analysis_json['softimpute_start'] = softimpute_start
    analysis_json['softimpute_end'] = softimpute_end
    analysis_json['softimpute_duration'] = softimpute_end - softimpute_start
    analysis_json['npsd_start'] = npsd_start
    analysis_json['npsd_end'] = npsd_end
    analysis_json['npsd_duration'] = npsd_end - npsd_start
    analysis_json['main_start'] = main_start
    analysis_json['main_end'] = main_end
    analysis_json['main_duration'] = main_end - main_start
    analysis_json['mean_squared_error'] = mse
    analysis_json['mean_squared_error_of_dropped_elements'] = msede
    analysis_json['mean_absolute_error'] = mae
    analysis_json['mean_absolute_error_of_dropped_elements'] = maede
    analysis_json['relative_error'] = re
    analysis_json['relative_error_of_dropped_elements'] = rede

    fd = open(completionanalysisfile, "w")
    json.dump(analysis_json, fd)
    fd.close()

if __name__ == "__main__":
    main()

