import sys

import numpy as np
import scipy as sp
from scipy import io
from scipy.io import wavfile
from scipy import signal

import plotly.offline as po
import plotly.graph_objs as pgo

def plot(file_name, similarities, files):
    # To fix the direction of the matrix as the diagonal line is from top-left to bottom-right.
    similarities_ = similarities[::-1]
    files_to_show = []
    for f in files:
        files_to_show.append(f.split('/')[-1].split('.')[0])
    files_to_show_ = files_to_show[::-1]
    
    data = [pgo.Heatmap(z=similarities_,
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
    filename = sys.argv[1]
    mat = io.loadmat(filename)
    similarities = mat['gram']
    files = mat['indices']

    filename_html = filename.replace('.mat', '.html')
    plot(filename_html, similarities, files)

if __name__ == "__main__":
    main()
