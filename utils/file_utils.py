import pickle

def save_new_result(filename, dataset_type, gram, sample_names):
    """ Saves the given results to the specified file.

    :param filename:
    :param dataset_type:
    :param gram:
    :param sample_names:
    :type filename: str
    :type dataset_type: str
    :type gram: np.ndarrays
    :type sample_names: list of strs
    """
    file = open(filename, 'wb')
    dic = dict(dataset_type=dataset_type,
               gram_matrices=[dict(gram_original=gram)],
               drop_indices=[[]],
               sample_names=sample_names,
               log=["made by GAK"])
    pickle.dump(dic, file)
    file.close()

def append_and_save_result(filename, prev_results, dropped, completed, completed_npsd, dropped_indices, action):
    """ Append and saves the given results to the specified file.

    :param filename:
    :param prev_results:
    :param dropped:
    :param completed:
    :param completed_npsd:
    :param dropped_indices:
    :param action:
    :type filename: str
    :type prev_results: dict
    :type dropped: np.ndarrays
    :type completed: np.ndarrays
    :type completed_npsd: np.ndarrays
    :type dropped_indices: list of ints
    :type action: str
    """
    gram_matrices = prev_results['gram_matrices']
    new_gram_matrices = dict(completed_npsd=completed_npsd,
                             completed=completed,
                             dropped=dropped)
    gram_matrices.append(new_gram_matrices)

    new_dropped_indices = prev_results['dropped_indices']
    new_dropped_indices.append(dropped_indices)

    log = prev_results['log']
    log.append(action)

    file = open(filename, 'wb')
    dic = dict(dataset_type=prev_results['dataset_type'],
               gram_matrices=gram_matrices,
               dropped_indices=new_dropped_indices,
               sample_names=prev_results['sample_names'],
               log=log)
    pickle.dump(dic, file)
    file.close()