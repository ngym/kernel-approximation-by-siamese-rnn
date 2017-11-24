import sys, os, shutil
import time

from sacred import Experiment

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from algorithms import gak
from datasets import others
from datasets.read_sequences import read_sequences, pick_labels
from utils import file_utils
from datasets.data_augmentation import augment_data, create_drop_flag_matrix

ex = Experiment('calculate_gram_matrix')

@ex.config
def cfg():
    #dataset_type: "6DMG", "6DMGupperChar", "upperChar", "UCIcharacter", "UCIauslan"
    output_dir = "results/"
    # Global Alignment Kernel Parameter
    sigma = None
    # Global Alignment Kernel Parameter
    triangular  = None
    # Location of dataset
    dataset_location = None
    # The format of the output file name
    output_file_format = "out"
    # You can save GRAM matrix in pickel format or hdf5.
    hdf5 = False
    # The ratio of the size of the number of sequences
    # after data augmentation.
    # 1 means 1 times, hence no augmentation is applied.
    data_augmentation_size = 1
    # For too large GRAM matrix because of data augmentation,
    # you can calculate only a part of kernel values
    # between a set of augmented data then approximate
    # other parts by matrix completion.
    drop_rate_between_augmenteds = 0
    # Parallelizm
    num_process = 4

@ex.named_config
def upperChar():
    dataset_type = "upperChar"

@ex.named_config
def UCIcharacter():
    dataset_type = "UCIcharacter"

@ex.named_config
def UCIauslan():
    dataset_type = "UCIauslan"

@ex.automain
def run(dataset_type, dataset_location, sigma, triangular, output_dir,
        output_filename_format, data_augmentation_size,
        drop_rate_between_augmenteds, num_process, hdf5):
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(os.path.abspath(sys.argv[2]),
                os.path.join(output_dir, os.path.basename(sys.argv[2])))

    assert others.is_valid_dataset_type(dataset_type)

    output_dir = os.path.abspath(output_dir)
    assert os.path.isdir(output_dir)

    seqs, key_to_str, _ = read_sequences(dataset_type, dataset_location)

    print("%d samples." % len(seqs))
    flag_augmented = [False] * len(seqs)
    
    if data_augmentation_size != 1:
        augmentation_magnification = 1.2
        seqs, key_to_str, flag_augmented = augment_data(seqs, key_to_str,
                                                        augmentation_magnification,
                                                        rand_uniform=True,
                                                        num_normaldist_ave=data_augmentation_size - 2)
    drop_flag_matrix = create_drop_flag_matrix(drop_rate_between_augmenteds,
                                               flag_augmented)
    
    sample_names = list(seqs.keys())

    start = time.time()
    gram = gak.gram_gak(list(seqs.values()), sigma, triangular,
                        num_process=num_process,
                        drop_flag_matrix=drop_flag_matrix)
    end = time.time()

    output_filename_format = output_filename_format.replace("${sigma}", str(sigma))\
                                                   .replace("${triangular}", str(triangular))
    if hdf5:
        log_file = os.path.join(output_dir, output_filename_format + ".hdf5")
        timelog = log_file.replace(".hdf5", ".timelog")
    else:
        log_file = os.path.join(output_dir, output_filename_format + ".pkl")
        timelog = log_file.replace(".pkl", ".timelog")
    file_utils.save_new_result(log_file, dataset_type, gram, sample_names, hdf5=hdf5)

    duration = end - start
    num_samples = len(sample_names)
    time_fd = open(timelog, 'w')
    time_fd.write("gram_gak_start: %d\n" % start)
    time_fd.write("gram_gak_end: %d\n" % end)
    time_fd.write("gram_gak_duration: %d\n" % duration)
    time_fd.write("num_samples: %d\n" % num_samples)
    time_fd.write("average_time_per_gak: %.5f\n" % (duration / (num_samples ** 2) * num_process))
    time_fd.close()


    
