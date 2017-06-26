import sys
import numpy as np
from scipy import io

from functools import reduce

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_gram_to_pdf(file_name, gram, files, separators, labels,
         sigma, drop_percent):
    """Plot Gram matrix and save as pdf file with matplotlib.
    
    :param file_name: Output pdf file name
    :param gram: Gram matrix to be plotted
    :param files: Filenames used for Global Alignment Kernel calcucation
    :param separators: Class sizes for label separators
    :param labels: Class names for labeling
    :param sigma: Global Alignment Kernel sigma parameter
    :param drop_percent: Amount of Gram matrix elements removed
    :type file_name: str
    :type gram: np.ndarray
    :type files: list of str
    :type separators: list of int
    :type labels: list of str
    :type sigma: float
    :type drop_percent: float
    """
             
    assert len(separators) == len(labels) - 1

    cmap = plt.get_cmap('bwr')
    cmap.set_bad('black')
    
    fig, ax = plt.subplots()
    f1 = ax.imshow(gram,
                   interpolation='nearest',
                   cmap=cmap,
                   vmax=1, vmin=0
                   )

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
    size = len(files) // len(labels) // 10
    separators_ = [0] + separators + [len(gram)]
    for i in range(len(labels)):
        label = labels[i]
        x = -len(files)//100
        y = np.mean([separators_[i], separators_[i+1]]) \
            + size
        #y = np.mean([separators_[i], separators_[i+1]]) \
        #    + len(files) // 75
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
                    rotation='vertical',
                    xy=(0,0), xytext=(x, y))

    # Title of the graph, displaying the sigma and the drop percent
    titletext = "σ =" + str(sigma) + "    drop=" + drop_percent #+ "%\n "
    ax.set_title(titletext,
                 horizontalalignment='center')

    #fig.suptitle(titletext,
    #             horizontalalignment='center')

    plt.savefig(file_name, format='pdf', dpi=1200)
    

def main():
    """Read .mat file, parse its metadata, plot Gram matrix and save as pdf file with matplotlib.
    """
    
    filename = sys.argv[1]
    mat = io.loadmat(filename)
    gram = mat['gram']
    files = mat['indices']
    seqs = {}

    if filename.find("audioset") != -1:
        i = 0
        while files[i].find("Bark") != -1:
            i += 1
            separators = [i]
        labels = ['Bark', 'Meow']
    elif filename.find("num") != -1:
        i = 0
        while files[i].find("num_3") != -1:
            i += 1
            separators = [i]
        labels = ['3', '4']
    elif filename.find("upperChar") != -1:
        labels = ["A", "B", "C", "D", "E", "F", "G",
                     "H", "I", "J", "K", "L", "M", "N",
                     "O", "P", "Q", "R", "S", "T", "U",
                     "V", "W", "X", "Y", "Z"]
        separators = []
        for alph in alphabets[:-1]:
            i = 0
            while files[i].find("upper_" + alph) == -1:
                i += 1
            while files[i].find("upper_" + alph) != -1:
                i += 1
            separators.append(i)
    elif filename.find("UCIcharacter") != -1:
        labels = []
        for f in files:
            l = f[0]
            if l not in labels:
                labels.append(l)
        separators = []
        for l in labels[:-1]:
            i = 0
            while files[i].split('/')[-1].find(l) == -1:
                i += 1
            while files[i].split('/')[-1].find(l) != -1:
                i += 1
            separators.append(i)
    elif filename.find("UCIauslan") != -1 or filename.find("UCItctodd") != -1:
        labels = []
        for f in files:
            l = reduce(lambda a, b: a + "-" + b, f.split('/')[-1].split('-')[:-2])
            if l not in labels:
                labels.append(l)
        separators = []
        for l in labels[:-1]:
            i = 0
            while reduce(lambda a, b: a + "-" + b, files[i].split('/')[-1].split('-')[:-2]) != l:
                i += 1
            while reduce(lambda a, b: a + "-" + b, files[i].split('/')[-1].split('-')[:-2]) == l:
                i += 1
            separators.append(i)
    else:
        assert False    
        
    sigma = float(filename.split("sigma")[1]
                  .split("_")[0]
                  .replace(".mat", ""))
    if filename.find("loss") == -1:
        drop_percent = "0%"
    else:
        drop_percent = filename.split("loss")[1].split("_")[0].replace(".mat", "") + "%"

    filename_pdf = filename.replace(".mat", ".pdf")
    
    # OUTPUT
    plot_gram_to_pdf(filename_pdf,
                     gram,
                     files,
                     separators,
                     labels,
                     sigma,
                     drop_percent)

    filename_pdf_dropped = filename.replace(".mat", "_dropped.pdf")
    plot_gram_to_pdf(filename_pdf_dropped,
                     mat['dropped_gram'],
                     files,
                     separators,
                     labels,
                     sigma,
                     drop_percent)
    
    filename_pdf_orig = filename.replace(".mat", "_orig.pdf")
    plot_gram_to_pdf(filename_pdf_orig,
                     mat['orig_gram'],
                     files,
                     separators,
                     labels,
                     sigma,
                     drop_percent)
    
    
if __name__ == "__main__":
    main()

