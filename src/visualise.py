
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.utils import unflatten


def rank_labels(pd_ser):
    '''
    rank behaviour variables and ignore labels of sparsed variables.
    return label and a flatten array of the current values
    '''
    pd_ser = pd_ser.replace(to_replace=0, value=np.nan)
    pd_ser = pd_ser.sort_values(ascending=False, )

    behav_labels = list(pd_ser.index)
    v_ranked = pd_ser.values
    v_ranked_flat = np.zeros((len(behav_labels),1))
    v_ranked_flat.flat[:v_ranked.shape[0]] = v_ranked    
    
    return v_ranked_flat, behav_labels

def plot_heatmap(ax, mat, x_labels, y_labels, cb_max, cmap=plt.cm.RdBu_r):
    '''
    plot one single genaric heatmap
    Only when axis is provided

    ax: the axis of figure
    mat: 2-d matrix
    x_labels, y_labels: lists of labels
    cb_max: maxium value of the color bar
    '''
    graph = ax.matshow(mat, vmin=-cb_max, vmax=cb_max, cmap=cmap)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(x_labels, rotation='vertical')
    ax.set_yticklabels(y_labels)
    return graph

def single_heatmap(mat, x_labels, y_labels, cb_label):
    '''
    heat map with color bar 
    '''
    cb_max = np.max(np.abs(mat))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    hm = ax.matshow(mat, vmin=-cb_max, vmax=cb_max, cmap=plt.cm.RdBu_r)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=1)
    cb = fig.colorbar(hm, cax=cax)
    cb.set_label(cb_label)

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    return fig

def plot_SCCA_FC_MWQ(FC_ws, behav_ws, region_labels, behav_labels, cb_max, cmap=plt.cm.RdBu_r):
    '''
    plotting tool for functional connectivity vs MRIQ
    '''
    plt.close('all')

    fig = plt.figure(figsize=(15,4))

    ax = fig.add_subplot(111)

    brain = plot_heatmap(ax, FC_ws, region_labels, region_labels, cb_max, cmap)
    # add a line to a diagnal
    ax.plot([-0.5, len(region_labels)-0.5], [-0.5, len(region_labels)-0.5], ls='--', c='.3')

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("right", size="1%", pad=8)
    behav = plot_heatmap(ax2, behav_ws, [' '], behav_labels, cb_max, cmap)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="50%", pad=0.25)
    fig.colorbar(brain, cax=cax)

    return fig


def map_labels(data, lables):
    df = pd.DataFrame(data, index=lables)
    return df

def show_results(u, v, u_labels, v_labels, rank_v=True, sparse=True):
    '''
    for plotting the scca decompostion heatmapt 
    u must be from a functional connectivity data set
    v must be from a data set that can be expressed in a single vector
    '''
    
    df_v = map_labels(v, v_labels)
    n_component = v.shape[1]

    # find maxmum for the color bar
    u_max = np.max(np.abs(u))
    v_max = np.max(np.abs(v))
    cb_max = np.max((u_max, v_max))
    
    figs = []

    for i in range(n_component):
    # reconstruct the correlation matrix
        ui = unflatten(u[:, i])

        if rank_v:
            vi, cur_v_labels = rank_labels(df_v.iloc[:, i])
            

        else:
            vi = v[:, i - 1 :i] # the input of the plot function must be an array
            cur_v_labels = v_labels
        
        if sparse:
            idx = np.isnan(vi).reshape((vi.shape[0]))
            vi = vi[~idx]
            vi = vi.reshape((vi.shape[0], 1))
            cur_v_labels = np.array(cur_v_labels)[~idx] 

        cur_fig = plot_SCCA_FC_MWQ(ui, vi, u_labels, cur_v_labels, cb_max=cb_max, cmap=plt.cm.RdBu_r)
        # save for later
        figs.append(cur_fig)
    return figs

from matplotlib.backends.backend_pdf import PdfPages

def write_pdf(fname, figures):
    '''
    write a list of figures to a single pdf
    '''
    doc = PdfPages(fname)
    for fig in figures:
        fig.savefig(doc, format='pdf', dpi=150, bbox_inches='tight')
    doc.close()

def write_png(fname, figures):
    '''
    write a list of figures to separate png files
    '''
    for i, fig in enumerate(figures):
        fig.savefig(fname.format(i + 1), dpi=150, bbox_inches='tight')

def set_text_size(size):
    '''
    set all the text in the figures 
    the font is always sans-serif. You only need this
    '''
    font = {'family' : 'sans-serif',
            'sans-serif' : 'Arial',
            'size' : size}
    matplotlib.rc('font', **font)

