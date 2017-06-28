import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json, pickle, time
from string import Template

import numpy as np
from scipy import io
import dill
from pathos.multiprocessing import ProcessingPool
from collections import OrderedDict

from utils.nearest_positive_semidefinite import nearest_positive_semidefinite
from utils.errors import mean_squared_error, mean_absolute_error, relative_error
from utils.plot_gram_to_html import plot_gram_to_html
from utils.make_matrix_incomplete import gram_drop_random, gram_drop_samples
from datasets.read_sequences import read_sequences
from utils.plot_gram_to_html import plot_gram_to_html
from algorithms.gak import gak, calculate_gak_sigma, calculate_gak_triangular

def gram_complete_gak(gram, seqs, indices, sigma=None, triangular=None):
    """Fill in multiple rows and columns of Gram matrix.

    :param gram: Gram matrix to be filled in
    :param seqs: List of time series to be used of filling in
    :param indices: Rows and columns to be filled in
    :param sigma: TGA kernel scale parameter
    :param triangular: TGA kernel band parameter
    :type gram: list of lists
    :type seqs: list of np.ndarrays    
    :type indices: list of ints
    :type sigma: float
    :type triangular: int
    :returns: Filled in version of Gram matrix
    :rtype: list of lists, list of tuples
    """

    if sigma is None:
        sigma = calculate_gak_sigma(seqs)
    if triangular is None:
        triangular = calculate_gak_triangular(seqs)

    pool = ProcessingPool()
    num_seqs = len(seqs)
    num_job = len(indices) * (num_seqs - len(indices)) + (len(indices) ** 2 - len(indices)) / 2
    num_finished_job = 0
    start_time = time.time()
    not_indices = list(set(range(num_seqs))-set(indices))
    for index in reversed(sorted(indices)):
        to_fill = [i for i in indices if i < index] + not_indices
        gram[index, to_fill] = pool.map(lambda j: gak(seqs[index], seqs[j], sigma, triangular), to_fill)
        gram[index, index] = 1.
        gram[to_fill, index] = gram[index, to_fill].T
        num_finished_job += len(to_fill)
        current_time = time.time()
        duration_time = current_time - start_time
        eta = duration_time * num_job / num_finished_job - duration_time
        print("[%d/%d], %ds, ETA:%ds" % (num_finished_job, num_job, duration_time, eta), end='\r')
    end_time = time.time()
    print("[%d/%d], %ds, ETA:%ds" % (num_finished_job, num_job, duration_time, eta))
    pool.close()
    return gram, start_time, end_time

def main():
    main_start = time.time()
    if len(sys.argv) != 2:
        random_drop = True
        gram_filename = sys.argv[1]
        if 'nipg' in os.uname().nodename:
            sample_dir = "~/shota/dataset"
        elif os.uname().nodename == 'atlasz' or 'cn' in os.uname().nodename:
            sample_dir = "/users/milacski/shota/dataset"
        elif os.uname().nodename == 'Regulus.local':
            sample_dir = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset"
        elif os.uname().nodename.split('.')[0] in {'procyon', 'pollux', 'capella',
                                                     'aldebaran', 'rigel'}:
            sample_dir = "/home/ngym/NFSshare/Lorincz_Lab/"
        else:
            sample_dir = sys.argv[4]
        drop_percent = int(sys.argv[2])
        completionanalysisfile = sys.argv[3]
        gak_sigma = float(sys.argv[5])
    else:
        random_drop = False
        config_json_file = sys.argv[1]
        config_dict = json.load(open(config_json_file, 'r'))

        gram_filename = config_dict['gram_file']
        sample_dir = config_dict['sample_dir']
        indices_to_drop = config_dict['indices_to_drop']
        completionanalysisfile = config_dict['completionanalysisfile']
        gak_sigma = config_dict['gak_sigma']

    fd = open(gram_filename, 'rb')
    pkl = pickle.load(fd)
    fd.close()
    
    dataset_type = pkl['dataset_type']
    gram_matrices = pkl['gram_matrices']
    if len(gram_matrices) == 1:
        gram = gram_matrices[0]['gram_original']
    else:
        gram = gram_matrices[-1]['gram_completed_npsd']
        
    #sample_names = [sn.replace(' ', '') for sn in pkl['sample_names']]
    sample_names = pkl['sample_names']

    logfile_loss = completionanalysisfile.replace(".timelog", ".losses")
    
    seqs = OrderedDict((k, v) for k, v in read_sequences(dataset_type, direc=sample_dir).items()
                       if k.split('/')[-1] in sample_names)
    
    seed = 1

    if random_drop:
        gram_drop, dropped_elements = gram_drop_random(seed, gram, drop_percent)
        logfile_html = gram_filename.replace(".pkl", "_drop" + str(drop_percent) + "_GAK_Completion.html")
        logfile_pkl  = gram_filename.replace(".pkl", "_drop" + str(drop_percent) + "_GAK_Completion.pkl")
    else:
        gram_drop, dropped_elements = gram_drop_samples(gram, indices_to_drop)
        logfile_html = gram_filename.replace(".pkl", "_dropfrom" + str(indices_to_drop[0]) + "_GAK_Completion.html")
        logfile_pkl  = gram_filename.replace(".pkl", "_dropfrom" + str(indices_to_drop[0]) + "_GAK_Completion.pkl")

    # GAK Completion
    #row_dropped = list(set([i[0] for i in indices_to_drop]))
    gram_completed, gak_start, gak_end\
        = gram_complete_gak(gram_drop, list(seqs.values()), indices_to_drop,
                            sigma=gak_sigma, triangular=None)
    gak_duration = gak_end - gak_start
    num_samples = len(seqs)
    
    gram_completed = np.array(gram_completed)
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

    print("GAK Completion files are output.")

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
    analysis_json['gak_start'] = gak_start
    analysis_json['gak_end'] = gak_end
    analysis_json['gak_duration'] = gak_end - gak_start
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

