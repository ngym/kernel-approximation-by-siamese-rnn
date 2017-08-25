import sys
from scipy import io

import plotly.offline as po
import plotly.graph_objs as pgo

from utils.file_utils import load_pickle

def plot_gram_to_html(file_name, gram, files):
    """Plot Gram matrix to html with plotly.
    
    :param file_name: Output html file name
    :param gram: Gram matrix to be plotted
    :param files: Filenames for labels
    :type file_name: str
    :type gram: np.ndarray
    :type files: list of str
    """

    # To fix the direction of the matrix as the diagonal line is from top-left to bottom-right.
    gram_ = gram[::-1]
    files_to_show = []
    for f in files:
        files_to_show.append(f.split('/')[-1].split('.')[0])
    files_to_show_ = files_to_show[::-1]
    
    data = [pgo.Heatmap(z=gram_,
                        x=files_to_show,
                        y=files_to_show_,
                        zmin=0, zmax=1)
            ]
    layout = pgo.Layout(xaxis=dict(side='top',
                                   tickfont=dict(size=5)),
                        yaxis=dict(tickfont=dict(size=5)))
    fig = pgo.Figure(data=data, layout=layout)
    po.plot(fig, filename=file_name, auto_open=False)

def main():
    """Read .mat file and plot Gram matrix to html with plotly.
    """
    
    filename = sys.argv[1]
    pkl = load_pickle(filename)
    dataset_type = pkl['dataset_type']
    gram_matrices = pkl['gram_matrices']
    sample_names = pkl['sample_names']

    matrices = gram_matrices[-1]
    for key in matrices.keys():
        filename_html = filename.replace('.pkl', "_" + key + '.html')
        plot_gram_to_html(filename_html, matrices[key], sample_names)

if __name__ == "__main__":
    main()

