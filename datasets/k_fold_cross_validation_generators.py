import pickle

""" k-fold cross-validation generators
"""

class KFold():
    """Base class for k-fold cross-validation test set generator.
    Puts ith sample into fold (i modulo fold count)
    I.e. each new sample goes to next fold
    Assumes that pkl['sample_names'] is sorted wrt classes

    :param pkl_file_path: .pkl file path for original Gram matrix
    :type pkl_file_path: str
    """
    
    def __init__(self, pkl_file_path):
        # assume pkl['sample_names'] is sorted with ground truth
        fd = open(pkl_file_path, 'rb')
        pkl = pickle.load(fd)
        self.sample_names = pkl['sample_names']
        self.generate_folds()
        
    def generate_folds(self):
        self.num_folds = 4
        self.fold = [[] for i in range(self.num_folds)]
        for i in range(len(self.sample_names)):
            self.fold[i % self.num_folds].append(i)

    def __iter__(self):
        self.k = 0
        return self

    def __next__(self):
        if self.k == len(self.fold):
            raise StopIteration()
        retval = self.fold[self.k]
        self.k += 1
        return retval    

class KFold_UCIauslan(KFold):
    """Class for k-fold cross-validation test set generator on UCI AUSLAN data set.
    This data set has 9 trials (recorded over 9 days) which defines a natural 9-fold separation.
 
    :param pkl_file_path: .pkl file path for original Gram matrix
    :type pkl_file_path: str
    """

    def __init__(self, pkl_file_path):
        super(KFold_UCIauslan, self).__init__(pkl_file_path)

    def generate_folds(self):
        self.num_folds = 5
        self.fold = [[] for i in range(self.num_folds)]
        for i in range(len(self.sample_names)):
            sample_name = self.sample_names[i]
            k = int(sample_name.split('/')[-2][-1])
            self.fold[(k - 1) // 2].append(i)

class KFold_6DMGupperChar(KFold):
    """
    :param pkl_file_path: .pkl file path for original Gram matrix
    :type pkl_file_path: str
    """

    def __init__(self, pkl_file_path):
        super(KFold_6DMGupperChar, self).__init__(pkl_file_path)

    def generate_folds(self):
        k_groups = [["A1", "C1", "C2", "C3", "C4"],
                    ["E1", "G1", "G2", "G3", "I1"],
                    ["I2", "I3", "J1", "J2", "J3"],
                    ["L1", "M1", "S1", "T1", "U1"],
                    ["Y1", "Y2", "Y3", "Z1", "Z2"]]
        self.num_folds = len(k_groups)
        self.fold = [[] for i in range(self.num_folds)]
        for i in range(len(self.sample_names)):
            sample_name = self.sample_names[i]
            type_, ground_truth, k_group, trial = sample_name.split('/')[-1].split('_')
            for k in range(len(k_groups)):
                if k_group in k_groups[k]:
                    self.fold[k].append(i)





