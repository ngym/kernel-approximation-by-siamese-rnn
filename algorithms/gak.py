import sys
import numpy as np

import global_align as ga


def gak(seq1, seq2, sigma, triangular):
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
    
    val = ga.tga_dissimilarity(seq1, seq2, sigma, triangular)
    kval = np.exp(-val)
    return kval

def main():
    seq1 = np.array(eval(sys.argv[1])).astype(np.double)
    seq2 = np.array(eval(sys.argv[2])).astype(np.double)

    val = gak(seq1, seq2, 0.4, 500)
    print(val)


if __name__ == "__main__":
    main()

