import numpy as np
import cvxpy

import sys, pickle
from svm_for_completed_testsets import train_validation_test_split, convert_index_to_attributes 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, roc_auc_score

def kernel_group_lasso(kDD, kDx, size_groups=None, l=0.01, max_iters=10000, eps=1e-4, gpu=False):
    """Solve Kernel Group LASSO problem via CVXPY with large scale SCS solver.

    :param kDD: Dictionary-vs-dictionary Gram matrix, positive semidefinite (d x d)
    :param kDx: Dictionary-vs-input Gram vector (d x 1)
    :param size_groups: List of size of groups
    :param l: Regularization parameter
    :param max_iters: Iteration count limit
    :param eps: Convergence tolerance
    :type kDD: np.ndarray
    :type kDx: np.ndarray
    :type num_groups: int
    :type l: float
    :type max_iters: int
    :type eps: float

    References:
        [1] `Jeni, László A., et al. "Spatio-temporal event classification using time-series kernel based structured sparsity."
        <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5098425/>`_.
    """

    
    assert(isinstance(kDD, np.ndarray) and kDD.ndim == 2 and kDD.shape[0] == kDD.shape[1]) #, (kDD.shape, kDx.shape))
    assert(isinstance(kDx, np.ndarray) and kDx.ndim == 1 and kDx.shape[0] == kDD.shape[0]) #, (kDD.shape, kDx.shape))
    if size_groups is None:
        size_groups = [1] * kDD.shape[0]
    assert(np.sum(size_groups) == len(kDx))
    assert(l >= 0)
    cumsum = np.cumsum(size_groups)
    alpha = cvxpy.Variable(kDD.shape[0])
    obj = cvxpy.Minimize( 0.5 * cvxpy.quad_form(alpha, kDD) \
                          - cvxpy.sum_entries(cvxpy.mul_elemwise(kDx, alpha)) \
                          + l * cvxpy.norm(cvxpy.vstack(*[np.sqrt(e - s) * cvxpy.norm(alpha[s:e]) for (s, e) in zip(np.concatenate([np.array([0]), cumsum[:-1]]), cumsum)]), 1)
    )

    prob = cvxpy.Problem(obj)
    prob.solve(solver='SCS', max_iters=max_iters, verbose=True, eps=eps, gpu=gpu)
    a = np.asarray(alpha.value)
    g = np.asarray([(1. / np.sqrt(e - s)) * np.linalg.norm(a[s:e]) for (s, e) in zip(np.concatenate([np.array([0]), cumsum[:-1]]), cumsum)])
    return a, g

def main():
    fd = open(sys.argv[1], 'rb')
    pkl = pickle.load(fd)
    fd.close()

    if len(pkl['gram_matrices']) == 1:
        gram = pkl['gram_matrices'][0]['gram_original']
    else:
        gram = pkl['gram_matrices'][-1]['gram_completed_npsd']
    num_indices = len(gram)
    dataset_type = pkl['dataset_type']
    indices = pkl['sample_names']

    test_indices = pkl['drop_indices']

    gtruths = [convert_index_to_attributes(index, dataset_type)['ground_truth'] for index in indices]
    
    test_matrix, _, _, train_matrix, train_gtruths, _, _, test_gtruths = train_validation_test_split(gram, gtruths, [], test_indices)

    size_groups = []
    for gt in list(set(train_gtruths)):
        size_groups.append(len(np.where(np.array(train_gtruths) == gt)[0]))

    pred_prob = []
    i = 0
    for test_vec in test_matrix:
        a, g = kernel_group_lasso(np.array(train_matrix), np.array(test_vec),
                                  size_groups=size_groups,
                                  l=0.01, max_iters=100, eps=1e-4, gpu=False)
        pred_prob.append(g)
        i += 1
        print("[%d/%d]" % (i, len(test_matrix)))

    pred = [list(sorted(set(gtruths)))[i] for i in np.argmax(np.array(pred_prob), axis=1)]        
    f1_ = f1_score(test_gtruths, pred, average='weighted')
    
    test_gtruths_binary_table = []
    for t in test_gtruths:
        tmp = [0] * len(list(set(train_gtruths)))
        index = list(sorted(set(train_gtruths))).index(t)
        tmp[index] = 1
        test_gtruths_binary_table.append(tmp)
    test_gtruths_binary_table = np.array(test_gtruths_binary_table)
    auc_ = roc_auc_score(y_true=test_gtruths_binary_table, y_score=np.array(pred_prob))
    print("f1_score: " + repr(f1_))
    print("roc_auc_score:" + repr(auc_))

if __name__ == '__main__':
    main()


