def kfolds(list, fold_count):
    """K-fold cross-validation test set generator.
    Puts ith element into fold (i modulo fold count)

    :param list: List.
    :param fold_count: Number of folds.
    :type list: list
    :type fold_count: int
    """
    folds = [[] for _ in range(fold_count)]
    for i in range(len(list)):
        folds[i % fold_count].append(i)
    return folds


def kfolds_UCIauslan(sample_names, fold_count):
    """K-fold cross-validation test set generator on UCI AUSLAN data set.
    This data set has 9 trials (recorded over 9 days) which defines a natural 9-fold separation

    :param sample_names: List of sample names.
    :param fold_count: Number of folds.
    :type sample_names: list of str
    :type fold_count: int
    """
    folds = [[] for _ in range(fold_count)]
    for i in range(len(sample_names)):
        k = int(sample_names[i].split('/')[-2][-1])
        folds[(k - 1) // 2].append(i)
    return folds


def kfold_6DMGupperChar(sample_names, fold_count):
    groups = ["A1", "C1", "C2", "C3", "C4",
              "E1", "G1", "G2", "G3", "I1",
              "I2", "I3", "J1", "J2", "J3",
              "L1", "M1", "S1", "T1", "U1",
              "Y1", "Y2", "Y3", "Z1", "Z2"]
    dic_list = [dict.fromkeys(groups[i:i + fold_count], i // fold_count)
                for i in range(0, len(groups), fold_count)]
    k_groups = {k: v for d in dic_list for k, v in d.items()}

    folds = [[] for _ in range(fold_count)]
    for i in range(len(sample_names)):
        try:
            type_, ground_truth, group, trial = sample_names[i].split('/')[-1].split('_')
        except:
            type_, ground_truth, group, trial, _, _ = sample_names[i].split('/')[-1].split('_')
        folds[k_groups[group]].append(i)
    return folds


def get_kfolds(dataset_type, sample_names, fold_count):
    if dataset_type == "UCIauslan":
        return kfolds_UCIauslan(sample_names, fold_count)
    if dataset_type in {"6DMG", "6DMGupperChar", "upperChar"}:
        return kfold_6DMGupperChar(sample_names, fold_count)
    return kfolds(sample_names, fold_count)
