import sys, os
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import file_utils


def plot_gram_to_pdf(file_name, gram, files,
                     separators, labels,
                     dataset_name, title="",
                     sigma=None, drop_percent=None,
                     rotate_vertically=True):
    """Plot Gram matrix and save as pdf file with matplotlib.

    :param file_name: Output pdf file name
    :param gram: Gram matrix to be plotted
    :param files: Filenames used for Global Alignment Kernel calcucation
    :param separators: Class sizes for label separators
    :param labels: Class names for labeling
    :param dataset_name: Name of the dataset
    :param title: Title of the plot
    :param sigma: Global Alignment Kernel sigma parameter
    :param drop_percent: Amount of Gram matrix elements removed
    :param rotate_vertically: If True then rotate the bottom labels vertically
    :type file_name: str
    :type gram: np.ndarray
    :type files: list of str
    :type separators: list of int
    :type labels: list of str
    :type dataset_name: str
    :type title: str
    :type sigma: float
    :type drop_percent: float
    :type rotate_vertically: bool
    """
    assert len(separators) == len(labels) - 1

    cmap = plt.get_cmap('bwr')
    cmap.set_bad('black')

    fig, ax = plt.subplots()
    f1 = ax.imshow(gram,
                   interpolation='nearest',
                   cmap=cmap,
                   vmax=1, vmin=0)

    # Colorbar on the right of the graph
    fig.colorbar(f1) #, ax=ax) #, shrink=0.9)

    # Separator of labels in the graph
    for sep in separators:
        # https://matplotlib.org/examples/color/colormaps_reference.html
        ax.axhline(linewidth=0.25, y=(sep),
                   color='black')#, ls='dashed')
        ax.axvline(linewidth=0.25, x=(sep),
                   color='black')#, ls='dashed')

    # Labels
    ## actually deleting all labels at first and annotate corresponding texts
    ## to corresponding place, the center of each group of data
    plt.axis('off')
    size = min(fig.get_size_inches() * fig.get_dpi() // len(labels) // 2)
    separators_ = [0] + separators + [len(gram)]
    rotation = 'vertical' if rotate_vertically else 'horizontal'
    for i in range(len(labels)):
        label = labels[i]
        x = -len(files)//100
        y = separators_[i] + 0.75 * (separators_[i+1] - separators_[i])
        if all([len(l) == 1 for l in labels]):
            ha = "center"
            x = -len(files)//50
        else:
            ha = "right"
        ax.annotate(label, horizontalalignment=ha,
                    size=size,
                    xy=(0,0), xytext=(x, y))
        x = np.mean([separators_[i], separators_[i+1]])
        y = len(files) + size * 2
        ax.annotate(label, horizontalalignment='center',
                    verticalalignment='top',
                    size=size,
                    rotation=rotation,
                    xy=(0, 0), xytext=(x, y))

    # Title of the graph, displaying the sigma and the drop percent
    titletext = dataset_name
    if title != "":
        titletext += ": " + title
    if sigma is not None:
        titletext += " σ =" + str(sigma)
    if drop_percent is not None:
        titletext += ", drop=" + str(drop_percent)
    ax.set_title(titletext,
                 horizontalalignment='center')

    plt.savefig(file_name, format='pdf', dpi=1200)


def get_informations(dataset_type, sample_names):
    def get_labels_and_separators(sample_names, dataset_type):
        """Get labels and separators from sample names.
        :param sample_names: A list of sample names.
        :param dataset_type: 6DMGupperChar or UCIauslan or UCIcharacter.
        :type sample_names: list of str
        :type dataset_type: str
        :return: List of labels and list of separators.
        :rtype: list of str, list of int
        """
        # Lambdas to calculate labels from keys of the sequences
        get_label = dict.fromkeys(["6DMG", "6DMGupperChar", "upperChar"], lambda fn: fn.split('/')[-1].split('_')[1])
        get_label["UCIcharacter"] = lambda str: str[0]
        get_label["UCIauslan"] = lambda fn: fn.split('/')[-1].split('-')[0]

        if dataset_type not in get_label:
            assert False

        label_to_separator = OrderedDict()
        for index, sample_name in enumerate(sample_names):
            label = get_label[dataset_type](sample_name)
            if label not in label_to_separator:
                label_to_separator[label] = index
        labels = list(label_to_separator.keys())
        separators = list(label_to_separator.values())[1:]
        return labels, separators

    dataset_name = dataset_type
    rotate = True
    if dataset_type in {"6DMG", "6DMGupperChar", "upperChar"}:
        dataset_name = "6DMG"
        rotate = False
    elif dataset_type == "UCIcharacter":
        rotate = False

    labels, separators = get_labels_and_separators(sample_names, dataset_type)

    return labels, separators, dataset_name, rotate


def main():
    """Read .pkl file, parse its metadata, plot Gram matrix and save as pdf file with matplotlib.
    """
    filename = os.path.abspath(sys.argv[1])
    title = sys.argv[2] if len(sys.argv) > 2 else ""

    assert filename[-4:] == ".pkl"

    pkl = file_utils.load_pickle(filename)
    dataset_type = pkl['dataset_type']
    gram_matrices = pkl['gram_matrices']
    sample_names = pkl['sample_names']

    labels, separators, dataset_name, rotate = get_informations(dataset_type, sample_names)

    if len(gram_matrices) == 1:
        matrices = gram_matrices[0]
    else:
        matrices = gram_matrices[-1]

    for key in matrices.keys():
        filename_pdf = filename.replace(".pkl", key + ".pdf")
        plot_title = title + " " + key.replace("_", " ")
        plot_gram_to_pdf(filename_pdf, matrices[key], sample_names,
                         separators, labels,
                         dataset_name, title=plot_title,
                         rotate_vertically=rotate)


if __name__ == "__main__":
    main()

