import unittest
import numpy as np
from scipy.stats import truncnorm

import datasets.data_augmentation as DA

class test_augmentation(unittest.TestCase):
    def setUp(self):
        pass
    def create_seq(self, length):
        seq = []
        for i in range(length):
            seq.append([float(i),float(i),float(i)])
        return np.array(seq)
    def test_augment_random(self):
        # lengthen
        length = 9000
        seq = self.create_seq(10)
        seq = DA.augment_random(seq, length)
        i = 1
        num_inserted = []
        count = 0
        print(seq.shape)
        for step in seq:
            if step.tolist() == [float(i), float(i), float(i)]:
                i += 1
                num_inserted.append(count)
                count = 0
            count += 1
        print(num_inserted)
        print(seq)


        # shorten
        length = 10
        seq = self.create_seq(10000)
        seq = DA.augment_random(seq, length)
        print(seq)


        counter = [0] * 10
        for i in range(100000):
            counter[np.random.randint(10)] += 1
        print(counter)
    def test_augment_normal_distribution(self):
        print("test_augment_normal_distribution")
        length = 9000
        seq = self.create_seq(10)
        seq = DA.augment_random(seq, length)
        i = 1
        num_inserted = []
        count = 0

        ave_ = 3
        std = 1
        seq = self.create_seq(100)
        seq = DA.augment_normal_distribution(seq, length, ave_, std)
        self.assertEqual(len(seq), length)
        i = 1
        num_inserted = []
        count = 0
        print(seq.shape)
        for step in seq:
            if step.tolist() == [float(i), float(i), float(i)]:
                i += 1
                num_inserted.append(count)
                count = 0
            count += 1
        print(num_inserted)
        print(seq)

        ave = 3
        std = 1
        minimum = 0
        maximum = 10
        counter = [0] * 10
        for i in range(100000):
            counter[int(truncnorm.rvs((minimum - ave) / std,
                                      (maximum - 1 - ave) / std,
                                      loc=ave,
                                      scale=std))] += 1
        print(counter)
    def tearDown(self):
        pass








if __name__ == '__main__':
    unittest.main()


    
