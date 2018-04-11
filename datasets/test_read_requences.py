import unittest
import datasets.read_sequences as rs
from collections import Counter

class TestReadSequences(unittest.TestCase):
    def setUp(self):
        self.dataset_type = "UCIarabicdigits"
        self.direc = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset/UCIarabicdigits"
    def test_read_sequences(self):
        seqs, key_to_str, _ = rs.read_sequences(self.dataset_type,
                                                direc=self.direc)
        labels = key_to_str.values()
        c = Counter(labels)
        print(c)
    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()



