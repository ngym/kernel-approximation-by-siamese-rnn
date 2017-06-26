import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json, pickle
from string import Template

import numpy as np
from scipy import io
import dill
from pathos.multiprocessing import ProcessingPool

import global_align as ga
from datasets.read_sequences import read_sequences
from utils.plot_gram_to_html import plot_gram_to_html

def gak(seq1, seq2, sigma=0.4, triangular=500):
    """Triangular Global Alignment (TGA) kernel computation between two time series.
    Meaningful similarity measure between time series
    Computes exponential soft minimum of all valid alignments
    May restrict the set of alignments with the triangular parameter
    Wrapper for the code of Marco Cuturi and Adrien Gaidon: http://marcocuturi.net/GA.html

    :param seq1: First time series
    :param seq2: Second time series
    :param sigma: TGA kernel scale parameter
    :param triangular: TGA kernel band parameter
    :type seq1: np.ndarray
    :type seq2: np.ndarray
    :type sigma: float
    :type triangular: int
    :returns: TGA kernel value
    :rtype: float
    """
    
    if seq1 is seq2:
        return 1
    
    seq1 = np.array(seq1).astype(np.double)
    seq2 = np.array(seq2).astype(np.double)
    val = ga.tga_dissimilarity(seq1, seq2, sigma, triangular)
    kval = np.exp(-val).astype(np.float32)
    return kval

def calculate_gak_sigma(seqs):
    """Calculate TGA sigma parameter setting.

    :param seqs: List of time series to be processed
    :type seqs: list of np.ndarrays
    :returns: Sigma
    :rtype: float
    """

    seq_ts = np.array([item for sublist in [[seq_t for seq_t in seq] for seq in seqs] for item in sublist])
    seq_ts_diff_norms = np.sqrt(np.sum(np.square(seq_ts[:, None, :] - seq_ts[None, :, :]), axis=-1))
    del seq_ts
    sigma = np.median(seq_ts_diff_norms) * np.median([len(seq) for seq in seqs]) * 5.
    del seq_ts_diff_norms
    return sigma

def calculate_gak_triangular(seqs):
    """Calculate TGA triangular parameter setting.

    :param seqs: List of time series to be processed
    :type seqs: list of np.ndarrays
    :returns: Triangular
    :rtype: int
    """

    triangular = np.median([len(seq) for seq in seqs]) * 0.5
    return triangular

def gram_gak(seqs, sigma=None, triangular=None):
    """TGA Gram matrix computation for a list of time series.

    :param seqs: List of time series to be processed
    :param sigma: TGA kernel scale parameter
    :param triangular: TGA kernel band parameter
    :type seqs: list of np.ndarrays
    :type sigma: float
    :type triangular: int
    :returns: TGA Gram matrix
    :rtype: np.ndarray
    """
    
    if sigma is None:
        sigma = calculate_gak_sigma(seqs)
    if triangular is None:
        triangular = calculate_gak_triangular(seqs)

    l = len(seqs)
    gram = -1 * np.ones((l, l), dtype=np.float32)
    
    pool = ProcessingPool()
    for i in reversed(range(l)):
        gram[i, :i] = pool.map(lambda j: gak(seqs[i], seqs[j], sigma, triangular), range(i)) 
        gram[i, i] = 1.
        gram[:i, i] = gram[i, :i].T
    pool.close()
    return gram

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
    for index in reversed(sorted(indices)):
        gram[index, :index] = pool.map(lambda j: gak(seqs[i], seqs[j], sigma, triangular), range(index))
        gram[index, index] = 1.
        gram[:i, i] = gram[i, :i].T
    pool.close()
    return gram

def main():
    if len(sys.argv) == 2:
        config_json_file = sys.argv[1]
        config_dict = json.load(open(config_json_file, 'r'))
        
        dataset_type = config_dict['dataset_type']
        output_dir = config_dict['output_dir']
        if 'gak_sigma' in config_dict.keys():
            gak_sigma = np.float32(config_dict['gak_sigma'])
        if 'gak_triangular' in config_dict.keys():
            gak_triangular = np.float32(config_dict['gak_triangular'])
        if dataset_type in {"6DMG", "6DMGupperChar", "upperChar"}:
            sample_glob_arg = config_dict['data_mat_files']
            if 'data_attribute_type' in config_dict.keys():
                data_attribute_type = config_dict['data_attribute_type']
        elif dataset_type == "UCIcharacter":
            sample_glob_arg = config_dict['data_mat_file']
        elif dataset_type == "UCIauslan":
            sample_glob_arg = config_dict['data_tsd_files']
        else:
            assert False
        output_filename_format = Template(config_dict['output_filename_format']).safe_substitute(
            dict(dataset_type=dataset_type,
                 data_attribute_type=data_attribute_type,
                 gak_sigma=("%.3f" % gak_sigma)))
        
        html = output_dir + output_filename_format.replace("${completion_alg}", "GAK") + ".html" 
        pkl = output_dir + output_filename_format.replace("${completion_alg}", "GAK") + ".pkl" 

        seqs = read_sequences(dataset_type, list_glob_arg=sample_glob_arg)
        sample_names = seqs.keys()
        
        gram = gram_gak(seqs.values(), sigma=gak_sigma)
        plot_gram_to_html(html,
                          gram.tolist(), sample_names)
        dic = {}
        dic['dataset_type'] = dataset_type
        dic['gram_matrices'] = [dict(gram_original=gram)]
        dic['drop_indices'] = []
        dic['sample_names'] = sample_names
        dic['log'] = ["made by GAK"]
        f = open(pkl, 'wb')
        pickle.dump(dic, f)
        f.close()
        
    else:
        seq1 = eval(sys.argv[1])
        seq2 = eval(sys.argv[2])
        if len(sys.argv) == 5:
            sigma = sys.argv[3]
            triangular = sys.argv[4]
        val = gak(seq1, seq2, sigma, triangular)
        print(val)

if __name__ == "__main__":
    main()

