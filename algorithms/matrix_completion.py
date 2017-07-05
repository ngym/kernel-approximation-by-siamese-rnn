import copy
import time

from pathos.multiprocessing import ProcessingPool
from fancyimpute import SoftImpute

from algorithms import gak


# TODO almost the same as in gak
def gak_matrix_completion(gram_drop, seqs, indices, sigma=None, triangular=None):
    """Fill in multiple rows and columns of Gram matrix.

    :param gram_drop: Gram matrix to be filled in
    :param seqs: List of time series to be used of filling in
    :param indices: Rows and columns to be filled in
    :param sigma: TGA kernel scale parameter
    :param triangular: TGA kernel band parameter
    :type gram_drop: np.ndarrays
    :type seqs: list of np.ndarrays
    :type indices: list of ints
    :type sigma: float
    :type triangular: int
    :returns: Filled in version of Gram matrix
    :rtype: list of lists, list of tuples
    """

    gram = copy.deepcopy(gram_drop)

    if sigma is None:
        sigma = gak.calculate_gak_sigma(seqs)
    if triangular is None:
        triangular = gak.calculate_gak_triangular(seqs)

    pool = ProcessingPool()
    num_seqs = len(seqs)
    num_job = len(indices) * (num_seqs - len(indices)) + (len(indices) ** 2 - len(indices)) / 2
    num_finished_job = 0
    start_time = time.time()
    not_indices = list(set(range(num_seqs)) - set(indices))
    for index in reversed(sorted(indices)):
        to_fill = [i for i in indices if i < index] + not_indices
        gram[index, to_fill] = pool.map(lambda j, i=index: gak.gak(seqs[i], seqs[j], sigma, triangular), to_fill)
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
