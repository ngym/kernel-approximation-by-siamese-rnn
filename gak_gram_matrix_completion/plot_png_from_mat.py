import sys
import numpy as np
import scipy as sp
from scipy import io

import plotly.plotly as py
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

    py.image.save_as(fig,
                     filename=file_name,
                     scale=10)

def main():
    filename = sys.argv[1]
    mat = io.loadmat(filename)
    similarities = mat['gram']
    files = mat['indices']
    seqs = {}

    seed = 1
        
    png_out = filename.replace(".mat", ".png")
    
    # OUTPUT
    plot(png_out,
         similarities, files)

if __name__ == "__main__":
    main()
