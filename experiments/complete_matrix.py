import sys, os, shutil, time
from collections import OrderedDict

import numpy as np

from sacred import Experiment

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from algorithms import matrix_completion
from datasets.read_sequences import read_sequences
from datasets import k_fold_cross_validation
from utils import errors
from utils import file_utils
from utils import nearest_positive_semidefinite
from utils import make_matrix_incomplete
from algorithms import KSS_unsupervised_alpha_prediction

from datasets.read_sequences import read_sequences, pick_labels
from datasets.data_augmentation import augment_data, create_drop_flag_matrix
from datasets.others import filter_samples

ex = Experiment('complete_matrix')


@ex.config
def cfg():
    output_dir = "results/"
    params = None
    # The ratio of the size of the number of sequences
    # after data augmentation.
    # 1 means 1 times, hence no augmentation is applied.
    data_augmentation_size = 1
    # Split the data by fold_count
    # and treat fold_to_drop-th fold as test
    fold_count = 5
    fold_to_drop = 1


@ex.named_config
def gak():
    algorithm = "gak"
    params = dict(sigma=20,
                  triangular=None)


@ex.named_config
def softimpute():
    algorithm = "softimpute"


@ex.named_config
def rnn():
    algorithm = "rnn"
    params = dict(
        # Number of passes over data set
        epochs=100,
        # Early Stopping parameter
        patience=2,
        # Used in continued training
        epoch_start_from=1,
        # Recurrent Layer type (Vanilla, LSTM or GRU)
        rnn="LSTM",
        # Recurrent layer sizes
        rnn_units=[10,10],
        # Dense layer sizes
        dense_units=[3],
        # Dropout probability
        dropout=0.3,
        # Flag to switch between Forward and Bidirectional RNN
        bidirectional=False,
        # Flag to switch Batch Normalization on/off
        batchnormalization=False,
        # RNN implementation (0: CPU, 2: GPU, 1: any)
        implementation=1,
        # Mode to run (train, load_pretrained, continue_train)
        mode="train", 
        loss_function='mse',
        # The weight of in-domain kernel approximation in loss function
        loss_weight_ratio=10.0,
        # Implementation of the head of the Siamese network
        # (dense, dot_product, weighted_dot_product)
        siamese_joint_method="weighted_dot_product",
        # Activation of the top of each branch of the Siemese network
        siamese_arms_activation="linear",
        # Already trained model file
        trained_modelfile_hdf5=None)

@ex.capture
def check_algorithm(algorithm):
    assert algorithm in {"gak", "softimpute", "knn", "iterativesvd", "rnn"}


@ex.capture
def check_fold(fold_count, fold_to_drop, hdf5):
    assert (0 <= fold_to_drop <= fold_count) or hdf5


@ex.capture
def check_params(algorithm, params):
    if algorithm == "gak":
        if 'sigma' not in params:  # TODO else
            params['sigma'] = None
        if 'triangular' not in params:  # TODO else
            params['triangular'] = None

@ex.capture
def check_pickle_format(result_):
    result = list(result_.keys())
    assert "dataset_type" in result \
           and "gram_matrices" in result \
           and "drop_indices" in result \
           and "sample_names" in result \
           and "log" in result


def calculate_errors(gram, gram_completed_npsd, dropped_elements):
    mse = errors.mean_squared_error(gram, gram_completed_npsd)
    msede = errors.mean_squared_error(gram,
                                      gram_completed_npsd,
                                      dropped_elements)

    mae = errors.mean_absolute_error(gram, gram_completed_npsd)
    maede = errors.mean_absolute_error(gram,
                                       gram_completed_npsd,
                                       dropped_elements)

    re = errors.relative_error(gram,
                               gram_completed_npsd)
    rede = errors.relative_error(gram,
                                 gram_completed_npsd,
                                 dropped_elements)

    return mse, msede, mae, maede, re, rede


@ex.automain
def run(pickle_or_hdf5_location, dataset_location, fold_count, fold_to_drop,
        algorithm, params, output_dir, output_filename_format, output_file,
        data_augmentation_size):
    os.makedirs(output_dir, exist_ok=True)
    try:
        shutil.copy(os.path.abspath(sys.argv[2]), os.path.join(output_dir, os.path.basename(sys.argv[2])))
    except shutil.SameFileError:
        pass
    hdf5 = pickle_or_hdf5_location[-4:] == "hdf5"
    check_fold(fold_count, fold_to_drop, hdf5)
    check_algorithm(algorithm)
    check_params(algorithm, params)

    pickle_or_hdf5_location = os.path.abspath(pickle_or_hdf5_location)
    dataset_location = os.path.abspath(dataset_location)
    output_dir = os.path.abspath(output_dir)
    assert os.path.isdir(output_dir)
    assert os.path.exists(pickle_or_hdf5_location)

    time_main_start = os.times()

    hdf5 = pickle_or_hdf5_location[-4:] == "hdf5"
    if hdf5:
        loaded_data = file_utils.load_hdf5(pickle_or_hdf5_location)
    else:
        loaded_data = file_utils.load_pickle(pickle_or_hdf5_location)
        check_pickle_format(loaded_data)

    dataset_type = loaded_data['dataset_type']
    if dataset_type == 'UCIauslan':
        sample_names = loaded_data['sample_names']
    else:
        sample_names = [s.split('/')[-1].split('.')[0] for s in loaded_data['sample_names']]
    gram_matrices = loaded_data['gram_matrices']
    if len(gram_matrices) == 1:
        gram = gram_matrices[0]['original']
    else:
        gram = gram_matrices[-1]['completed_npsd']

    # drop elements
    if fold_count == 0:        
        gram_drop = gram
    else:
        folds = k_fold_cross_validation.get_kfolds(dataset_type, sample_names, fold_count)
        indices_to_drop = folds[fold_to_drop - 1]
        gram_drop, dropped_elements = make_matrix_incomplete.gram_drop_samples(gram, indices_to_drop)

    seqs, key_to_str, _ = read_sequences(dataset_type, dataset_location)
    seqs = filter_samples(seqs, sample_names)
    key_to_str = filter_samples(key_to_str, sample_names)
    
    if data_augmentation_size > 1:
        augmentation_magnification = 1.2
        seqs, key_to_str, flag_augmented = augment_data(seqs, key_to_str,
                                                        augmentation_magnification,
                                                        rand_uniform=True,
                                                        num_normaldist_ave=data_augmentation_size - 2)

    train_start = None
    train_end = None
    if algorithm == "gak":
        gram_completed, time_completion_start, time_completion_end \
            = matrix_completion.gak_matrix_completion(gram_drop, list(seqs.values()), indices_to_drop,
                                                      sigma=params['sigma'], triangular=params['triangular'])
        action = "GAK sigma: " + str(params['sigma']) + " triangular: " + str(params['triangular'])
        output_filename_format = output_filename_format.replace("${sigma}", str(params['sigma']))\
                                                       .replace("${triangular}", str(params['triangular']))
    elif algorithm in {"softimpute", "knn", "iterativesvd"}:
        if algorithm == "softimpute":
            func = matrix_completion.softimpute_matrix_completion
            action = "Softimpute"
            print('running SoftImpute')
        elif algorithm == "knn":
            func = matrix_completion.knn_matrix_completion
            action = "KNN"
            print('running KNN')
        elif algorithm == "iterativesvd":
            func = matrix_completion.iterativesvd_matrix_completion
            action = "IterativeSVD"
            print('running IterativeSVD')
        else:
            print("unsupported fancyimpute algorithm")
            exit(-1)
        flag_test = np.zeros(len(seqs))
        flag_test[indices_to_drop] = 1
        drop_flag_matrix = create_drop_flag_matrix(1 - params['gak_rate'],
                                                   flag_test, condition_or=True)
        for i in range(len(seqs)):
            drop_flag_matrix[i, i] = 1
            for j in range(i + 1):
                if i not in indices_to_drop and j not in indices_to_drop:
                    drop_flag_matrix[i, j] = 1
                    drop_flag_matrix[j, i] = 1

        print(len(seqs)**2)
        print(np.count_nonzero(drop_flag_matrix))
        gram_completed, time_completion_start, time_completion_end \
            = func(gram_drop,
                   list(seqs.values()),
                   sigma=params['sigma'],
                   triangular=params['triangular'],
                   num_process=params['num_process'],
                   drop_flag_matrix=drop_flag_matrix)
    elif algorithm == "rnn":
        modelfile_hdf5 = os.path.join(output_dir, output_filename_format + "_model.hdf5")
        logfile_loss = os.path.join(output_dir, output_filename_format + ".losses")
        gram_completed, time_train_start, time_train_end, time_completion_start, time_completion_end \
            = matrix_completion.rnn_matrix_completion(gram_drop,
                                                      list(seqs.values()),
                                                      params['epochs'],
                                                      params['patience'],
                                                      params['epoch_start_from'],
                                                      logfile_loss,
                                                      modelfile_hdf5,
                                                      params['rnn'],
                                                      params['rnn_units'],
                                                      params['dense_units'],
                                                      params['dropout'],
                                                      params['implementation'],
                                                      params['bidirectional'],
                                                      params['batchnormalization'],
                                                      params['mode'],
                                                      params['loss_function'],
                                                      params['loss_weight_ratio'],
                                                      list(key_to_str.values()),
                                                      params['siamese_joint_method'],
                                                      params['siamese_arms_activation'],
                                                      trained_modelfile_hdf5=params['trained_modelfile_hdf5'])
        action = "SiameseRNN"
    else:
        assert False

    # eigenvalue check
    time_npsd_start = os.times()
    gram_completed_npsd = nearest_positive_semidefinite.nearest_positive_semidefinite(gram_completed)
    time_npsd_end = os.times()

    # save results
    if hdf5:
        log_file = os.path.join(output_dir, output_filename_format + ".hdf5")
    else:
        log_file = os.path.join(output_dir, output_filename_format + ".pkl")
    action += " " + time.asctime(time.localtime())
    file_utils.append_and_save_result(log_file, loaded_data, gram_drop, gram_completed, gram_completed_npsd, indices_to_drop,
                                      action, hdf5=hdf5)

    # claculate errors
    mse, mse_dropped, mae, mae_dropped, re, re_dropped = calculate_errors(gram, gram_completed_npsd, dropped_elements)

    time_main_end = os.times()

    # save run times and errors
    num_calculated_elements = len(dropped_elements) - len(indices_to_drop) // 2
    num_dropped_sequences = len(indices_to_drop)
    out_path = os.path.join(output_dir, output_file)
    file_utils.save_analysis(out_path, len(dropped_elements),
                             num_dropped_sequences,
                             num_calculated_elements,
                             time_completion_start, time_completion_end,
                             time_npsd_start, time_npsd_end,
                             time_main_start, time_main_end,
                             mse, mse_dropped, mae, mae_dropped, re, re_dropped,
                             time_train_start=time_train_start, time_train_end=time_train_end)



    
