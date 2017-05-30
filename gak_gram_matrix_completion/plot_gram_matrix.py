import plotly.offline as po
import plotly.graph_objs as pgo

def plot(file_name, similarities, files):
    # To fix the direction of the matrix as the diagonal line is from top-left to bottom-right.
    similarities_ = similarities[::-1]
    files_to_show = []
    for f in files:
        files_to_show.append(f.split('/')[-1].split('.')[0])
    files_to_show_ = files_to_show[::-1]
    
    trace = pgo.Heatmap(z=similarities_,
                        x=files_to_show,
                        y=files_to_show_,
                        zmin=0, zmax=1
    )
    data=[trace]
    po.plot(data, filename=file_name, auto_open=False)

