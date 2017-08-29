import unittest
import numpy as np
import utils.file_utils as fu

class TestFileUtilHDF5(unittest.TestCase):
    def setUp(self):
        length = 6501 * 8
        arr = np.arange(length ** 2)
        arr.resize((length, length))
        name_string = "name_string"
        self.dic = dict(dic=dict(arr=arr), name_string=name_string)
    def test_save_and_load_hdf5(self):
        filename = "test.hdf5"
        fu.save_hdf5(filename, self.dic)
        dic = fu.load_hdf5(filename)
        self.__test_save_and_load_hdf5_rec(dic, self.dic)
    def __test_save_and_load_hdf5_rec(self, dic_loaded, dic_orig):
        for k in dic_loaded:
            if isinstance(dic_loaded[k], np.ndarray):
                self.assertTrue(np.all(dic_loaded[k] == dic_orig[k]))
            elif isinstance(dic_loaded[k], dict):
                self.assertTrue(isinstance(dic_orig[k], dict))
                self.__test_save_and_load_hdf5_rec(dic_loaded[k], dic_orig[k])
            else:
                self.assertEqual(dic_loaded[k], dic_orig[k])
    def tearDown(self):
        pass








if __name__ == "__main__":
    unittest.main()


    

    
