import time

import numpy as np
from pathos.multiprocessing import ProcessingPool

import algorithms.global_align as ga

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

def gram_gak(seqs, sigma=None, triangular=None, drop_rate=0, nodes=4):
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

    parallelism = 10000
    num_finished_job = 0
    current_time = time.time()
    list_duration_time = []
    num_eta_calculation_resource = 5
    
    jobs_gen = jobs_generator(l, parallelism, drop_rate)
    num_job = (l + 1) * l / 2
    num_job = int(num_job * (1 - drop_rate))
    pool = ProcessingPool(nodes=nodes)
    for current_jobs in jobs_gen:
        result_current_jobs = pool.map(lambda tup: (tup[0], tup[1], gak(seqs[tup[0]], seqs[tup[1]], sigma, triangular)), current_jobs)
        for i, j, gak_value in result_current_jobs:
            gram[i, j] = gak_value
            
        num_finished_job += len(current_jobs)

        prev_time = current_time
        current_time = time.time()
        duration_time = current_time - prev_time
        list_duration_time.append(duration_time)

        running_time = sum(list_duration_time)
        recent_running_time = sum(list_duration_time[-num_eta_calculation_resource:])
        num_involved_jobs_in_recent_running_time = min(len(list_duration_time),
                                                       num_eta_calculation_resource) * parallelism
        eta = recent_running_time * (num_job - num_finished_job) / num_involved_jobs_in_recent_running_time

        print("[%d/%d], %ds, ETA:%ds                             " % \
              (num_finished_job, num_job, running_time, eta), end='\r')
    pool.close()
    print("[%d/%d], %ds" % (num_finished_job, num_job, running_time))


    for i in len(gram):
        for j in len(gram[0]):
            if gram[i][j] == -1:
                gram[i][j] = np.nan
    
    return gram

def jobs_generator(l, parallelism, drop_rate):
    jobs = []
    for i in range(l):
        for j in range(i):
            if np.random.rand() < drop_rate:
                continue
            jobs.append((i,j))
            if len(jobs) == parallelism:
                yield jobs
                jobs = []
    yield jobs

def worker()    




    
