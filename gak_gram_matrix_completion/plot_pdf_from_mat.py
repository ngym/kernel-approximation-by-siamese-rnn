import sys
import numpy as np
import scipy as sp
from scipy import io

import matplotlib.pyplot as plt

def plot(file_name, similarities, files, separators, labels,
         sigma, loss):
    assert len(separators) == len(labels) - 1
    
    similarities_ = similarities[::-1]
    files_to_show = []
    for f in files:
        files_to_show.append(f.split('/')[-1].split('.')[0])
    files_to_show_ = files_to_show[::-1]

    fig, ax = plt.subplots()
    f1 = ax.imshow(similarities,
                   interpolation='nearest',
                   cmap='bwr',
                   )
    
    fig.colorbar(f1 )#, ax=ax) #, shrink=0.9)

    # separator of labels in the graph
    for sep in separators:
        # https://matplotlib.org/examples/color/colormaps_reference.html
        ax.axhline(linewidth=0.5, y=(sep - 0.5),
                   color='black', ls='dashed')
        ax.axvline(linewidth=0.5, x=(sep),
                   color='black', ls='dashed')


    plt.axis('off')
    separators_ = [0] + separators + [len(similarities)]
    for i in range(len(labels)):
        label = labels[i]
        y = np.mean([separators_[i], separators_[i+1]]) \
            + len(files) // 75
        ax.annotate(label, horizontalalignment='right',
                    xy=(0,0), xytext=(-len(files)//100, y))

        
    #legend = ax.legend(loc='upper', shadow=True,
    #                   fontsize='x-large')

    titletext = "σ =" + str(sigma) + "    loss=" + str(loss) + "%"
    ax.set_title(titletext,
                 horizontalalignment='center')

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
        sigma = 0.4
    elif filename.find("num") != -1:
        i = 0
        while files[i].find("num_3") != -1:
            i += 1
            separators = [i]
        labels = ['3', '4']
        sigma = 6.0
    else:
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
        sigma = 6.0
        
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
