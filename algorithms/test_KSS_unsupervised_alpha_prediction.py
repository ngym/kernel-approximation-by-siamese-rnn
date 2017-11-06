import os, sys
from collections import Counter
import numpy as np
import keras.backend as K

import unittest
import KSS_unsupervised_alpha_prediction as KU
from utils import file_utils
from utils import make_matrix_incomplete
from datasets.read_sequences import read_sequences
from datasets.others import filter_samples

class TestKss_Loss(unittest.TestCase):
    def setUp(self):
        pickle_or_hdf5_location = "results/6DMG/30/t1/gram_upperChar_sigma30_triangularNone_t1_noaugmentation.hdf5"
        dataset_location = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset/6DMG_mat_112712/matR_char"
        
        loaded_data = file_utils.load_hdf5(os.path.abspath(pickle_or_hdf5_location))
        gram_matrices = loaded_data['gram_matrices']
        self.gram = gram_matrices[0]['original']
        self.sample_names = loaded_data['sample_names']
        self.lmbd = 0.5
        
        dataset_type = loaded_data['dataset_type']
        sample_names = [s.split('/')[-1].split('.')[0] for s in loaded_data['sample_names']]
        seqs, key_to_str, _ = read_sequences(dataset_type, direc=dataset_location)
        seqs = filter_samples(seqs, sample_names)
        key_to_str = filter_samples(key_to_str, self.sample_names)
        labels = list(key_to_str.values())
        tmp = list(labels)
        counter = Counter(tmp)
        #self.size_groups = [counter[label] for label in sorted(set(tmp), key=tmp.index)]
        self.size_groups = [15] * 26
        
    def runTest(self):
        self.test_call()
    def test_call(self):
        len_gram = int(len(self.gram) * 0.6)
        print("len_seqs: %d" % len(self.gram))
        print("len_gram: %d" % len_gram)
        index_gram = list(range(len_gram))
        index_ks = list(range(len_gram, len(self.gram)))
        gram = self.gram[index_gram][:, index_gram]
        ks = self.gram[index_ks][:, index_gram]
        
        self.kl = KU.KSS_Loss(self.lmbd, gram, self.size_groups)
        alpha = np.array([list(range(len_gram)) for _ in range(len(index_ks))])
        alpha = alpha.astype(np.float32)
        assert(ks.shape == alpha.shape)
        self.kl(K.variable(ks), K.variable(alpha))
        
        
        
        
    def tearDown(self):
        pass



if __name__ == "__main__":
    t = TestKss_Loss()
    t.debug()
    #unittest.main()

    
    
