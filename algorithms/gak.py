import sys, time

import numpy as np
import multiprocessing as mp
import concurrent.futures

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

def gram_gak(seqs, sigma=None, triangular=None,
             num_process=4,
             drop_rate_between_augmenteds=0,
             flag_augmented=None):
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

    num_seq = len(seqs)
    gram = -1 * np.ones((num_seq, num_seq), dtype=np.float32)

    num_gak_per_job = 10000
    num_finished_gak = 0
    start_time = time.time()
    """
    current_time = time.time()
    list_duration_time = []
    num_eta_calculation_resource = 5
    """

    job_gen = job_generator(num_seq, num_gak_per_job,
                            drop_rate_between_augmenteds, flag_augmented)
    num_gak = (num_seq + 1) * num_seq / 2
    if drop_rate_between_augmenteds != 0:
        num_augmented_seqs = sum(flag_augmented)
        num_original_seqs = num_seq - num_augmented_seqs
        num_gak_between_originals = (num_original_seqs + 1) * num_original_seqs / 2
        num_gak_between_original_and_augmented = num_original_seqs * num_augmented_seqs
        num_gak_between_augmenteds = num_gak - num_gak_between_originals\
                                     - num_gak_between_original_and_augmented
        num_gak_between_augmenteds = int(num_gak_between_augmenteds * \
                                         (1 - drop_rate_between_augmenteds))
        num_gak = num_gak_between_originals +\
                  num_gak_between_original_and_augmented +\
                  num_gak_between_augmenteds

    print("using %d multi processes." % num_process)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_process) as executor:
        first_jobs = []
        for i in range(num_process + 1):
            try:
                first_jobs.append(next(job_gen))
            except StopIteration:
                pass
        print("Start submitting jobs.")
        futures = [executor.submit(worker, job, seqs, sigma, triangular)
                   for job in first_jobs]
        while futures != []:
            for future in concurrent.futures.as_completed(futures):
                futures.remove(future)
                worker_result = future.result()
                for i, j, gak_value in worker_result:
                    gram[i, j] = gak_value
                    gram[j, i] = gak_value
                try:
                    job = next(job_gen)
                    futures.append(executor.submit(worker, job, seqs, sigma, triangular))
                except StopIteration:
                    pass

                num_finished_gak += len(worker_result)

                running_time = time.time() - start_time
                eta = running_time / num_finished_gak\
                      * (num_gak - num_finished_gak)

                sec = eta % 60
                minu = (eta // 60) % 60
                hour = (eta // (60 * 60)) % 24
                day = eta // (60 * 60 * 24)
                print("[%d/%d], %ds, ETA:%dd:%dh:%dm:%ds" % (num_finished_gak, num_gak,
                                                             running_time,
                                                             day, hour, minu, sec)\
                      + " " * 30, end='\r')
    print("[%d/%d], %ds" % (num_finished_gak, num_gak, running_time) + " " * 30)

    for i in range(len(gram)):
        for j in range(len(gram[0])):
            if gram[i][j] == -1:
                gram[i][j] = np.nan
    
    return gram

def job_generator(l, num_gak_per_job,
                  drop_rate_between_augmenteds,
                  flag_augmented):
    if drop_rate_between_augmenteds != 0:
        assert flag_augmented != None
        assert len(flag_augmented) == l
    job = []
    for i in range(l):
        for j in range(i + 1):
            if flag_augmented != None:
                if flag_augmented[i] and flag_augmented[j]:
                    if np.random.rand() < drop_rate_between_augmenteds:
                        continue
            job.append((i, j))
            if len(job) == num_gak_per_job:
                yield job
                job = []
    yield job

def worker(job, seqs, sigma, triangular):
    result_job = []
    for i, j in job:
        result_job.append((i, j, gak(seqs[i], seqs[j], sigma, triangular)))
    return result_job
