import sys, os
import time

from sacred import Experiment

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.read_sequences import read_sequences
from algorithms import gak
from utils import file_utils

ex = Experiment('calculate_gram_matrix')

@ex.config
def cfg():
    #dataset_type: "6DMG", "6DMGupperChar", "upperChar", "UCIcharacter", "UCIauslan"
    output_dir = "results/"
    sigma = None
    triangular  = None

@ex.named_config
def upperChar():
    dataset_type = "upperChar"

@ex.named_config
def UCIcharacter():
    dataset_type = "UCIcharacter"

@ex.named_config
def UCIauslan():
    dataset_type = "UCIauslan"

@ex.capture
def check_dataset_type(dataset_type):
    assert dataset_type in {"6DMG", "6DMGupperChar", "upperChar", "UCIcharacter", "UCIauslan"}

@ex.automain
def run(dataset_type, dataset_location, sigma, triangular, output_dir, output_filename_format):
    check_dataset_type(dataset_type)

    seqs, _, _ = read_sequences(dataset_type, direc=dataset_location)
    sample_names = list(seqs.keys())

    start = time.time()
    gram = gak.gram_gak(list(seqs.values()), sigma, triangular)
    end = time.time()

    output_filename_format = output_filename_format.replace("${sigma}", str(sigma))\
                                                   .replace("${triangular}", str(triangular))
    
    log_file = os.path.join(output_dir, output_filename_format + ".pkl")
    file_utils.save_new_result(log_file, dataset_type, gram, sample_names)

    timelog = log_file.replace(".pkl", ".timelog")
    duration = end - start
    num_samples = len(sample_names)
    time_fd = open(timelog, 'w')
    time_fd.write("gram_gak_start: %d\n" % start)
    time_fd.write("gram_gak_end: %d\n" % end)
    time_fd.write("gram_gak_duration: %d\n" % duration)
    time_fd.write("num_samples: %d\n" % num_samples)
    time_fd.write("average_time_per_gak: %.5f\n" % (duration / (num_samples ** 2)))
    time_fd.close()