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
            seq.append([float(i), float(i)])
        return np.array(seq)
    def test_insert_steps_with_balance(self):
        seq = self.create_seq(3)
        seq = DA.insert_steps_with_balance(seq,
                                           [1,1,1,0,0,0])
        seq_ = np.array([[0, 0],
                         [0.25, 0.25],
                         [0.5, 0.5],
                         [0.75, 0.75],
                         [1, 1],
                         [1.25, 1.25],
                         [1.5, 1.5],
                         [1.75, 1.75],
                         [2, 2]])
        self.assertTrue(np.all(seq == seq_))

        
    def test_insert_steps_between_two_with_balance(self):
        seq = self.create_seq(3)
        seq = DA.insert_steps_between_two_with_balance(seq,
                                                       1,
                                                       3)
        seq_ = np.array([[0, 0],
                         [1, 1],
                         [1.25, 1.25],
                         [1.5, 1.5],
                         [1.75, 1.75],
                         [2, 2]])
        self.assertTrue(np.all(seq == seq_))
    def test_insert_random(self):
        # lengthen
        length = 9000
        seq = self.create_seq(10)
        seq = DA.insert_random(seq, length)
        
        i = 1
        num_inserted = []
        count = 0
        print(seq.shape)
        for step in seq:
            if step.tolist() == [float(i), float(i)]:
                i += 1
                num_inserted.append(count - 1)
                count = 0
            count += 1
        print(num_inserted)
        print(seq)
    def test_insert_normal_distribution(self):
        # lengthen
        length = 9000
        for ave_p in {0.25, 0.50, 0.75}:
            seq = self.create_seq(10)
            ave = (seq.shape[0] - 2) * ave_p
            std = seq.shape[0] * 0.25
            seq = DA.insert_normal_distribution(seq, length,
                                                ave, std)
            i = 1
            num_inserted = []
            count = 0
            print(seq.shape)
            for step in seq:
                if step.tolist() == [float(i), float(i)]:
                    i += 1
                    num_inserted.append(count - 1)
                    count = 0
                count += 1
            print(num_inserted)
            print(seq)
        """
    def test
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
    """
    def tearDown(self):
        pass








if __name__ == '__main__':
    unittest.main()


    
