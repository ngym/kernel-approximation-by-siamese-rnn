import json

import pickle
import h5py

def load_pickle(filename):
    file = open(filename, 'rb')
    pkl = pickle.load(file)
    file.close()
    return pkl


def save_pickle(filename, dic):
    file = open(filename, 'wb')
    pickle.dump(dic, file)
    file.close()


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
    dic = dict(dataset_type=dataset_type,
               gram_matrices=[dict(original=gram)],
               dropped_indices=[[]],
               sample_names=sample_names,
               log=["made by GAK"])
    save_pickle(filename, dic)


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

    dic = dict(dataset_type=prev_results['dataset_type'],
               gram_matrices=gram_matrices,
               dropped_indices=new_dropped_indices,
               sample_names=prev_results['sample_names'],
               log=log)
    save_pickle(filename, dic)


def save_json(filename, dic):
    file = open(filename, "w")
    json.dump(dic, file)
    file.close()


def save_analysis(filename, drop_count, calculated_count,
                  completion_start, completion_end, npsd_start, npsd_end, main_start, main_end,
                  mse, mse_dropped, mae, mae_dropped, re, re_dropped,
                  train_start=None, train_end=None):
    analysis_json = {}
    analysis_json['number_of_dropped_elements'] = drop_count
    analysis_json['number_of_calculated_elements'] = calculated_count
    if train_start is not None and train_end is not None:
        analysis_json['train_start'] = train_start
        analysis_json['train_end'] = train_end
        analysis_json['train_duration'] = train_end - train_start
    analysis_json['completion_start'] = completion_start
    analysis_json['completion_end'] = completion_end
    analysis_json['completion_duration'] = completion_end - completion_start
    analysis_json['npsd_start'] = npsd_start
    analysis_json['npsd_end'] = npsd_end
    analysis_json['npsd_duration'] = npsd_end - npsd_start
    analysis_json['main_start'] = main_start
    analysis_json['main_end'] = main_end
    analysis_json['main_duration'] = main_end - main_start
    analysis_json['mean_squared_error'] = mse
    analysis_json['mean_squared_error_of_dropped_elements'] = mse_dropped
    analysis_json['mean_absolute_error'] = mae
    analysis_json['mean_absolute_error_of_dropped_elements'] = mae_dropped
    analysis_json['relative_error'] = re
    analysis_json['relative_error_of_dropped_elements'] = re_dropped

    save_json(filename, analysis_json)





##################
#      HDF5      #
##################

def load_hdf5(filename):
    dic = {}
    with h5py.File(filename, 'r') as hdf5_value:
        for k, v in hdf5_value.items():
            dic[k] = load_hdf5_rec(v)
    return dic

def load_hdf5_rec(hdf5_obj):
    if isinstance(hdf5_obj, h5py.Group):
        dic = {}
        for k, v in hdf5_obj.items():
            dic[k] = load_hdf5_rec(v)
        return dic
    elif isinstance(hdf5_obj, h5py.Dataset):
        return hdf5_obj.value
    else:
        assert False

def save_hdf5(filename, dic):
    file = h5py.File(filename, 'w')
    for k, v in dic.items():
        save_hdf5_rec(file, k, v)
    file.flush()
    file.close()

def save_hdf5_rec(hdf5_obj, key, dic_or_value):
    if isinstance(dic_or_value, dict):
        hdf5_obj.create_group(key)
        for k, v in dic_or_value.items():
            save_hdf5_rec(hdf5_obj[key], k, v)
    else:
        hdf5_obj.create_dataset(key, data=dic_or_value)



    
