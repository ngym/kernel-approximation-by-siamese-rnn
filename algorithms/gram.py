import time

import numpy as np
from pathos.multiprocessing import ProcessingPool

import algorithms.global_align as ga

class Gram:
    def __init__(self, sequences, sigma=None, triangular=None):
        """TGA Gram matrix computation for a list of time series.

        :param sequences: List of time series to be processed
        :param sigma: TGA kernel scale parameter
        :param triangular: TGA kernel band parameter
        :type sequences: list of np.ndarrays
        :type sigma: float
        :type triangular: int
        """
        self.sequences = sequences
        if sigma is not None:
            self.sigma = sigma
        else:
            self.sigma = self.__calculate_sigma()
        if triangular is not None:
            self.triangular = triangular
        else:
            self.triangular = self.__calculate_triangular()

    def __calculate_sigma(self):
        """Calculate TGA sigma parameter setting.

        :returns: Sigma
        :rtype: float
        """
        # TODO MemoryError
        time_series = np.array([item for sublist in [[seq_t for seq_t in seq] for seq in self.sequences] for item in sublist])
        diff_norms = np.sqrt(np.sum(np.square(time_series[:, None, :] - time_series[None, :, :]), axis=-1))
        sigma = np.median(diff_norms) * np.median([len(seq) for seq in self.sequences]) * 5.0
        return sigma

    def __calculate_triangular(self):
        """Calculate TGA triangular parameter setting.

        :returns: Triangular
        :rtype: int
        """
        return np.median([len(seq) for seq in self.sequences]) * 0.5

    def __calculate_tga_kernel(self, i, j):
        """Triangular Global Alignment (TGA) kernel computation between two time series.
        Meaningful similarity measure between time series
        Computes exponential soft minimum of all valid alignments
        May restrict the set of alignments with the triangular parameter
        Wrapper for the code of Marco Cuturi and Adrien Gaidon: http://marcocuturi.net/GA.html

        :param seq1: Index of the first time series
        :param seq2: Index of the second time series
        :type seq1: int
        :type seq2: int
        :returns: TGA kernel value
        :rtype: float
        """
        if i == j:
            return 1

        seq1 = np.array(self.sequences[i]).astype(np.double)
        seq2 = np.array(self.sequences[j]).astype(np.double)
        val = ga.tga_dissimilarity(seq1, seq2, self.sigma, self.triangular)
        kernel_val = np.exp(-val).astype(np.float32)
        return kernel_val

    def compute(self):
        """TGA Gram matrix computation.

        :returns: TGA Gram matrix
        :rtype: np.ndarray
        """
        l = len(self.sequences)
        gram = -1 * np.ones((l, l), dtype=np.float32)

        start_time = time.time()
        num_job = (1 + l) * l / 2
        num_finished_job = 0
        duration_time = 0
        eta = 0
        pool = ProcessingPool()
        for i in reversed(range(l)):
            gram[i, :i] = pool.map(lambda j: self.__calculate_tga_kernel(i, j), range(i))
            gram[i, i] = 1.
            gram[:i, i] = gram[i, :i].T
            num_finished_job = (i + l) * (l - i) / 2
            current_time = time.time()
            duration_time = current_time - start_time
            eta = duration_time * num_job / num_finished_job - duration_time
            print("[%d/%d], %ds, ETA:%ds" % (num_finished_job, num_job, duration_time, eta), end='\r')
        pool.close()
        print("[%d/%d], %ds, ETA:%ds" % (num_finished_job, num_job, duration_time, eta))
        return gram