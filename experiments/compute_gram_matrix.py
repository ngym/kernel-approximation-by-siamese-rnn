import sys, os

from sacred import Experiment

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.read_sequences import read_sequences
from algorithms.gram import Gram

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
def run(dataset_type, dataset_location, sigma, triangular):
    check_dataset_type(dataset_type)
    dataset, _, _ = read_sequences(dataset_type, dataset_location)
    gram = Gram(list(dataset.values()), sigma, triangular)
    return gram.compute()