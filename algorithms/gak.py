import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

import global_align as ga
from datasets.read_sequences import read_sequences

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
        seq_ts = np.array([item for sublist in [[seq_t for seq_t in seq] for seq in seqs] for item in sublist])
        seq_ts_diff_norms = np.sqrt(np.sum(np.square(seq_ts[:, None, :] - seq_ts[None, :, :]), axis=-1))
        sigma = np.median(seq_ts_diff_norms) * np.median([len(seq) for seq in seqs]) * 5.
    if triangular is None:
        triangular = np.median([len(seq) for seq in seqs]) * 0.5

    l = len(seqs)
    gram = -1 * np.ones((l, l), dtype=np.float32)
    for i in range(l):
        gram[i, i] = 1.
        for j in range(i + 1, l):
            gram[i, j] = gak(seqs[i], seqs[j], sigma, triangular)
        gram[i, :i] = gram[:i, i].T
    return gram

def main():
    if len(sys.argv) == 1:
        config_json_file = sys.argv[1]
        config_dict = json.load(open(config_json_file, 'r'))
        
        dataset_type = config_dict['dataset_type']
        sample_dir = config_dict['sample_dir']
        if 'gak_sigma' in config_dict.keys():
            gak_sigma = np.float32(config_dict['gak_sigma'])
        if 'gak_triangular' in config_dict.keys():
            gak_triangular = np.float32(config_dict['gak_triangular'])
        output_dir = config_dict['output_dir']

        html = output_dir + output_filename_format.replace("${completion_alg}", "GAK") + ".html" 
        mat = output_dir + output_filename_format.replace("${completion_alg}", "GAK") + ".mat" 

        seqs, seq_names = read_sequences(dataset_type, sample_dir)
        gram_grak(seqs)
        plot_html_gram(html,
                       gram.tolist(), seq_names)
        io.savemat(mat, dict(gram=gram.tolist(), seq_names=seq_names))
        
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

