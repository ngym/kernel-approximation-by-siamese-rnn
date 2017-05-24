import unittest
import svm

class Test_separate_gram(unittest.TestCase):
    def setUp(self):
        self.mat = []
        for i in range(10):
            row = []
            for j in range(10):
                row.append((i,j))
                row.append((i,j))
            self.mat.append(row.copy())
            self.mat.append(row.copy())

        self.data_attributes = []
        for i in range(10):
            self.data_attributes.append(dict(k_group=i))
            self.data_attributes.append(dict(k_group=i))
    def test_separate_gram(self):
        for k_group in range(10):
            matched, unmatched = svm.separate_gram(self.mat,
                                             self.data_attributes,
                                                   k_group)
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
