import sys
import numpy as np
import scipy as sp
from scipy import io

from functools import reduce

import matplotlib.pyplot as plt

def plot(file_name, similarities, files, separators, labels,
         sigma, loss):
    assert len(separators) == len(labels) - 1
    
    fig, ax = plt.subplots()
    f1 = ax.imshow(similarities,
                   interpolation='nearest',
                   cmap='bwr',
                   )

    # Colorbar on the right of the graph
    fig.colorbar(f1) #, ax=ax) #, shrink=0.9)

    # Separator of labels in the graph
    for sep in separators:
        # https://matplotlib.org/examples/color/colormaps_reference.html
        ax.axhline(linewidth=0.5, y=(sep - 0.5),
                   color='black', ls='dashed')
        ax.axvline(linewidth=0.5, x=(sep),
                   color='black', ls='dashed')

    # Labels
    ## actually deleting all labels at first and annotate corresponding texts
    ## to corresponding place, the center of each group of data
    plt.axis('off')
    separators_ = [0] + separators + [len(similarities)]
    for i in range(len(labels)):
        label = labels[i]
        x = -len(files)//100
        y = np.mean([separators_[i], separators_[i+1]]) \
            + len(files) // 75
        if all([len(l) == 1 for l in labels]):
            ha = "center"
            x = -len(files)//50
        else:
            ha = "right"
        ax.annotate(label, horizontalalignment=ha,
                    xy=(0,0), xytext=(x, y))
        x = np.mean([separators_[i], separators_[i+1]]) 
        y = -len(files)//100
        ax.annotate(label, horizontalalignment='center',
                    xy=(0,0), xytext=(x, y))

    # Title of the graph, displaying the sigma and the loss percentage
    titletext = "σ =" + str(sigma) + "    loss=" + str(loss) + "%\n "
    ax.set_title(titletext,
                 horizontalalignment='center')

    #fig.suptitle(titletext,
    #             horizontalalignment='center')

    plt.savefig(file_name, format='pdf', dpi=1200)
    

def main():
    filename = sys.argv[1]
    mat = io.loadmat(filename)
    similarities = mat['gram']
    files = mat['indices']
    seqs = {}

    if filename.find("audioset") != -1:
        i = 0
        while files[i].find("Bark") != -1:
            i += 1
            separators = [i]
        labels = ['Bark', 'Meow']
        sigma = float(filename.split("sigma")[1]
                      .split("_")[0]
                      .replace(".mat", ""))
    elif filename.find("num") != -1:
        i = 0
        while files[i].find("num_3") != -1:
            i += 1
            separators = [i]
        labels = ['3', '4']
        sigma = float(filename.split("sigma")[1]
                      .split("_")[0]
                      .replace(".mat", ""))
    elif filename.find("upperChar") != -1:
        alphabets = ["A", "B", "C", "D", "E", "F", "G",
                     "H", "I", "J", "K", "L", "M", "N",
                     "O", "P", "Q", "R", "S", "T", "U",
                     "V", "W", "X", "Y", "Z"]
        separators = []
        labels = []
        for alph in alphabets[:-1]:
            i = 0
            while files[i].find("upper_" + alph) == -1:
                i += 1
            while files[i].find("upper_" + alph) != -1:
                i += 1
            separators.append(i)
        labels = alphabets
        sigma = float(filename.split("sigma")[1]
                      .split("_")[0]
                      .replace(".mat", ""))
        print(sigma)
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
        sigma = float(filename.split("sigma")[1]
                      .split("_")[0]
                      .replace(".mat", ""))
    elif filename.find("UCItctodd") != -1:
        labels = []
        for f in files:
            l = reduce(lambda a, b: a + "-" + b, f.split('/')[-1].split('-')[:-2])
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
        sigma = float(filename.split("sigma")[1]
                      .split("_")[0]
                      .replace(".mat", ""))
    else:
        assert False    
        
    pdf_out = filename.replace(".mat", ".pdf")
    
    loss = 0
    # OUTPUT
    plot(pdf_out,
         similarities,
         files,
         separators,
         labels,
         sigma,
         loss)

if __name__ == "__main__":
    main()
