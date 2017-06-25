import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import unittest
from algorithms.svm import separate_gram

class Test_separate_gram(unittest.TestCase):
    """Unit test for Gram matrix cross-validation splitting.
    """
    
    def setUp(self):
        self.mat = []
        for i in range(10):
            row = []
            for j in range(10):
                row.append((i,j))
            for j in range(10):
                row.append((i,j))
            self.mat.append(row.copy())
        for i in range(10):
            row = []
            for j in range(10):
                row.append((i,j))
            for j in range(10):
                row.append((i,j))
            self.mat.append(row.copy())

        self.data_attributes = []
        for i in range(10):
            self.data_attributes.append(dict(k_group=i))
        for i in range(10):
            self.data_attributes.append(dict(k_group=i))
    def test_separate_gram(self):
        for k_group in range(10):
            matched, unmatched = separate_gram(self.mat, self.data_attributes, k_group)
            print(unmatched.__len__())
            print(unmatched[0].__len__())
            self.assertEqual(unmatched.__len__(), 18)
            self.assertEqual(unmatched[0].__len__(), 18)
            for unmatched_row in unmatched:
                for c in unmatched_row:
                    self.assertFalse(k_group in c)
            for matched_row in matched:
                for c in matched_row:
                    self.assertTrue(k_group in c)
            print("\n\n")
            print("k_group:" + str(k_group))
            print("unmatched")
            for r in unmatched:
                print(r)
            print("matched")
            for r in matched:
                print(r)
    def tearDown(self):
        pass
        
if __name__ == '__main__':
    unittest.main()

