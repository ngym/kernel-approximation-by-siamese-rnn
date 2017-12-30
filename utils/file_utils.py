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


def save_new_result(filename, dataset_type, gram, sample_names,
                    hdf5=False):
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
    if hdf5:
        print("save result in hdf5.")
        save_hdf5(filename, dic)
    else:
        print("save result in pickle.")
        save_pickle(filename, dic)


def append_and_save_result(filename, prev_results, dropped, completed, completed_npsd,
                           dropped_indices, action, hdf5=False):
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
    if hdf5:
        save_hdf5(filename, dic)
    else:
        save_pickle(filename, dic)

def save_json(filename, dic):
    file = open(filename, "w")
    json.dump(dic, file, indent=4, sort_keys=True)
    file.close()

def save_analysis(filename, drop_count,
                  num_calculated_sequences,
                  num_calculated_elements,
                  time_completion_start, time_completion_end,
                  time_npsd_start, time_npsd_end,
                  time_main_start, time_main_end,
                  mse, mse_dropped, mae, mae_dropped, re, re_dropped,
                  time_train_start=None, time_train_end=None):

    virtual_completion_duration = (time_completion_end.user + time_completion_end.system) - \
                                  (time_completion_start.user + time_completion_start.system)
    elapsed_completion_duration = time_completion_end.elapsed - time_completion_start.elapsed
    
    virtual_npsd_duration = (time_npsd_end.user + time_npsd_end.system) - (time_npsd_start.user + time_npsd_start.system)
    elapsed_npsd_duration = time_npsd_end.elapsed - time_npsd_start.elapsed

    virtual_main_duration = (time_main_end.user + time_main_end.system) - (time_main_start.user + time_main_start.system)
    elapsed_main_duration = time_main_end.elapsed - time_main_start.elapsed

    analysis_json = {}
    
    analysis_json['basics'] = {}
    analysis_json['basics']['number_of_calculated_sequences'] = num_calculated_sequences
    analysis_json['basics']['number_of_dropped_elements'] = drop_count
    analysis_json['basics']['number_of_calculated_elements'] = num_calculated_elements
    if time_train_start is not None and time_train_end is not None:
        analysis_json['basics']['train_duration'] = time_train_end.elapsed - time_train_start.elapsed
    analysis_json['basics']['mean_squared_error'] = mse
    analysis_json['basics']['mean_squared_error_of_dropped_elements'] = mse_dropped
    analysis_json['basics']['mean_absolute_error'] = mae
    analysis_json['basics']['mean_absolute_error_of_dropped_elements'] = mae_dropped
    analysis_json['basics']['relative_error'] = re
    analysis_json['basics']['relative_error_of_dropped_elements'] = re_dropped
    
    analysis_json['all'] = {}
    analysis_json['all']['virtual_completion_duration'] = virtual_completion_duration
    analysis_json['all']['elapsed_completion_duration'] = elapsed_completion_duration
    analysis_json['all']['virtual_npsd_duration'] = virtual_npsd_duration
    analysis_json['all']['elapsed_npsd_duration'] = elapsed_npsd_duration
    analysis_json['all']['virtual_completion_npsd_duration'] = virtual_completion_duration + virtual_npsd_duration
    analysis_json['all']['elapsed_completion_npsd_duration'] = elapsed_completion_duration + elapsed_npsd_duration
    #analysis_json['all']['virtual_main_duration'] = virtual_main_duration
    #analysis_json['all']['elapsed_main_duration'] = elapsed_main_duration
    
    analysis_json['each_elem'] = {}
    analysis_json['each_elem']['virtual_completion_duration_per_calculated_element'] = virtual_completion_duration / num_calculated_elements
    analysis_json['each_elem']['elapsed_completion_duration_per_calculated_element'] = elapsed_completion_duration / num_calculated_elements
    analysis_json['each_elem']['virtual_npsd_duration_per_calculated_element'] = virtual_npsd_duration / num_calculated_elements
    analysis_json['each_elem']['elapsed_npsd_duration_per_calculated_element'] = elapsed_npsd_duration / num_calculated_elements
    analysis_json['each_elem']['virtual_completion_npsd_duration_per_calculated_element'] = (virtual_completion_duration + virtual_npsd_duration) / num_calculated_elements
    analysis_json['each_elem']['elapsed_completion_npsd_duration_per_calculated_element'] = (elapsed_completion_duration + elapsed_npsd_duration) / num_calculated_elements
    #analysis_json['each_elem']['virtual_main_duration_per_calculated_element'] = virtual_main_duration / num_calculated_elements
    #analysis_json['each_elem']['elapsed_main_duration_per_calculated_element'] = elapsed_main_duration / num_calculated_elements

    analysis_json['each_seq'] = {}
    analysis_json['each_seq']['virtual_completion_duration_per_calculated_sequence'] = virtual_completion_duration / num_calculated_sequences
    analysis_json['each_seq']['elapsed_completion_duration_per_calculated_sequence'] = elapsed_completion_duration / num_calculated_sequences
    analysis_json['each_seq']['virtual_npsd_duration_per_calculated_sequence'] = virtual_npsd_duration / num_calculated_sequences
    analysis_json['each_seq']['elapsed_npsd_duration_per_calculated_sequence'] = elapsed_npsd_duration / num_calculated_sequences
    analysis_json['each_seq']['virtual_completion_npsd_duration_per_calculated_sequence'] = (virtual_completion_duration + virtual_npsd_duration) / num_calculated_sequences
    analysis_json['each_seq']['elapsed_completion_npsd_duration_per_calculated_sequence'] = (elapsed_completion_duration + elapsed_npsd_duration) / num_calculated_sequences
    #analysis_json['each_seq']['virtual_main_duration_per_calculated_sequence'] = virtual_main_duration / num_calculated_sequences
    #analysis_json['each_seq']['elapsed_main_duration_per_calculated_sequence'] = elapsed_main_duration / num_calculated_sequences

    dic=dict(prediction=analysis_json,
             classification=None)
    save_json(filename, dic)





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
        if all([k.isdigit() for k in hdf5_obj.keys()]):
            if sorted([int(k) for k in hdf5_obj.keys()]) == list(range(len(hdf5_obj))):
                # list
                lis = [None] * len(hdf5_obj)
                for k, v in hdf5_obj.items():
                    lis[int(k)] = load_hdf5_rec(v)
                return lis
        # dict
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
    elif isinstance(dic_or_value, list):
        hdf5_obj.create_group(key)
        for i in range(len(dic_or_value)):
            save_hdf5_rec(hdf5_obj[key], str(i), dic_or_value[i])
    else:
        hdf5_obj.create_dataset(key, data=dic_or_value)



    
