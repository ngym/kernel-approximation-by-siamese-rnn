import unittest
import datasets.read_sequences as rs

class Test_read_sequences(unittest.TestCase):
    def setUp(self):
        self.dataset_type = 'upperChar'
        dataset_location = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset/6DMG_mat_112712/matR_char"
        self.seqs, _, _ = rs.read_sequences(self.dataset_type, direc=dataset_location)
    def test_pick_labels(self):
        labels_to_use = ["I", "J"]
        seqs = rs.pick_labels(self.dataset_type, self.seqs, labels_to_use)
        print(seqs.keys())
        for k in seqs.keys():
            self.assertTrue(k[:7] in {"upper_I", "upper_J"})
    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()


    
