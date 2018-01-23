import copy
import os, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.stats.mstats import zscore

# loa my modules
from src.utils import load_pkl
from src.visualise import *

import joblib
import pickle

# Built-in modules #
import random

# Third party modules #
import numpy, scipy, matplotlib, pandas
from matplotlib import pyplot
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
import names

model_path = './models/SCCA_Yeo7nodes_revision_4_0.80_0.50.pkl'
label_path = '~/yeo7_cluster/names.csv'
dat_path = './data/processed/dict_SCCA_data_prepro_revision1.pkl'

# load data
model = joblib.load(model_path)
dataset = load_pkl(dat_path)
df_label = pd.read_csv(label_path)

#df = pd.read_pickle(df_path)
u, v = model.u * [1, 1, -1, 1] , model.v * [1, 1, -1, 1]
n = model.n_components

# create labels for the nodes
seed_names = df_label.iloc[:, 0].apply(str) + '-' + df_label.iloc[:, -2] + '-' + df_label.iloc[:, -3] + ' ' + df_label.iloc[:, -1]

# unflatten the functional corr coeff
from src.utils import unflatten

mat = []               
for i in range(4):
    mat.append(unflatten(u[:, i]))

# plot size
MY_FIX_SIZE = (13, 10)
###############################################################################
# Create Custom Color Gradients #
red_black_sky     = {'red':   ((0.0, 0.0, 0.0), (0.5, 0.0, 0.1), (1.0, 1.0, 1.0)),
                     'green': ((0.0, 0.0, 0.9), (0.5, 0.1, 0.0), (1.0, 0.0, 0.0)),
                     'blue':  ((0.0, 0.0, 1.0), (0.5, 0.1, 0.0), (1.0, 0.0, 0.0))}
red_black_blue    = {'red':   ((0.0, 0.0, 0.0), (0.5, 0.0, 0.1), (1.0, 1.0, 1.0)),
                     'green': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                     'blue':  ((0.0, 0.0, 1.0), (0.5, 0.1, 0.0), (1.0, 0.0, 0.0))}
red_black_green   = {'red':   ((0.0, 0.0, 0.0), (0.5, 0.0, 0.1), (1.0, 1.0, 1.0)),
                     'blue':  ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                     'green': ((0.0, 0.0, 1.0), (0.5, 0.1, 0.0), (1.0, 0.0, 0.0))}
yellow_black_blue = {'red':   ((0.0, 0.0, 0.0), (0.5, 0.0, 0.1), (1.0, 1.0, 1.0)),
                     'green': ((0.0, 0.0, 0.8), (0.5, 0.1, 0.0), (1.0, 1.0, 1.0)),
                     'blue':  ((0.0, 0.0, 1.0), (0.5, 0.1, 0.0), (1.0, 0.0, 0.0))}

make_cmap = lambda x: matplotlib.colors.LinearSegmentedColormap('my_colormap', x, 256)
color_gradients = {'red_black_sky'      : make_cmap(red_black_sky),
                   'red_black_blue'     : make_cmap(red_black_blue),
                   'red_black_green'    : make_cmap(red_black_green),
                   'yellow_black_blue'  : make_cmap(yellow_black_blue),
                   'red_white_blue'     : pyplot.cm.bwr,
                   'seismic'            : pyplot.cm.seismic,
                   'green_white_purple' : pyplot.cm.PiYG_r,
                   'coolwarm'           : pyplot.cm.coolwarm,}

###############################################################################
def plot_hierarchical_heatmap_core(frame):
    """A common use case for biologists analyzing their gene expression data is to cluster and visualize patterns of expression in the form of a heatmap and associated dendrogram."""

    row_method     = 'single'     # Can be: linkage, single, complete, average, weighted, centroid, median, ward
    column_method  = 'single'     # Can be: linkage, single, complete, average, weighted, centroid, median, ward
    row_metric     = 'braycurtis' # Can be: see scipy documentation
    column_metric  = 'braycurtis' # Can be: see scipy documentation
    gradient_span  = 'min_to_max_centered'   # Can be: min_to_max, min_to_max_centered, only_max, only_min
    color_gradient = 'coolwarm'   # Can be: see color_gradients dictionary
    fig_weight = MY_FIX_SIZE[0]
    fig_height = MY_FIX_SIZE[1]

    # Names #
    row_header = frame.index
    column_header = frame.columns

    # What color to use #
    cmap = color_gradients[color_gradient]

    # Scale the max and min colors #
    value_min = frame.min().min()
    value_max = frame.max().max()
    if gradient_span == 'min_to_max_centered':
        value_max = max([value_max, abs(value_min)])
        value_min = value_max * -1
    if gradient_span == 'only_max': value_min = 0
    if gradient_span == 'only_min': value_max = 0
    norm = matplotlib.colors.Normalize(value_min, value_max)

    # Scale the figure window size #
    fig = pyplot.figure(figsize=(fig_weight, fig_height))

    # Calculate positions for all elements #
    # ax1, placement of dendrogram 1, on the left of the heatmap
    ### The second value controls the position of the matrix relative to the bottom of the view
    [ax1_x, ax1_y, ax1_w, ax1_h] = [0.05, 0.22, 0.2, 0.6]
    width_between_ax1_axr = 0.004
    ### distance between the top color bar axis and the matrix
    height_between_ax1_axc = 0.004
    ### Sufficient size to show
    color_bar_w = 0.015

    # axr, placement of row side colorbar #
    ### second to last controls the width of the side color bar - 0.015 when showing
    [axr_x, axr_y, axr_w, axr_h] = [0.31, 0.1, color_bar_w, 0.6]
    axr_x = ax1_x + ax1_w + width_between_ax1_axr
    axr_y = ax1_y; axr_h = ax1_h
    width_between_axr_axm = 0.004

    # axc, placement of column side colorbar #
    ### last one controls the hight of the top color bar - 0.015 when showing
    [axc_x, axc_y, axc_w, axc_h] = [0.4, 0.63, 0.5, color_bar_w]
    axc_x = axr_x + axr_w + width_between_axr_axm
    axc_y = ax1_y + ax1_h + height_between_ax1_axc
    height_between_axc_ax2 = 0.004

    # axm, placement of heatmap for the data matrix #
    [axm_x, axm_y, axm_w, axm_h] = [0.4, 0.9, 2.5, 0.5]
    axm_x = axr_x + axr_w + width_between_axr_axm
    axm_y = ax1_y; axm_h = ax1_h
    axm_w = axc_w

    # ax2, placement of dendrogram 2, on the top of the heatmap #
    ### last one controls hight of the dendrogram
    [ax2_x, ax2_y, ax2_w, ax2_h] = [0.3, 0.72, 0.6, 0.15]
    ax2_x = axr_x + axr_w + width_between_axr_axm
    ax2_y = ax1_y + ax1_h + height_between_ax1_axc + axc_h + height_between_axc_ax2
    ax2_w = axc_w

    # axcb - placement of the color legend #
    [axcb_x, axcb_y, axcb_w, axcb_h] = [0.07, 0.88, 0.18, 0.09]

    # Compute and plot top dendrogram #
    if column_method:
        d2 = dist.pdist(frame.transpose())
        D2 = dist.squareform(d2)
        ax2 = fig.add_axes([ax2_x, ax2_y, ax2_w, ax2_h], frame_on=True)
        Y2 = sch.linkage(D2, method=column_method, metric=column_metric)
        Z2 = sch.dendrogram(Y2)
        ind2 = sch.fcluster(Y2, 0.7*max(Y2[:,2]), 'distance')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ### apply the clustering for the array-dendrograms to the actual matrix data
        idx2 = Z2['leaves']
        frame = frame.iloc[:,idx2]
        ### reorder the flat cluster to match the order of the leaves the dendrogram
        ind2 = ind2[idx2]
    else: idx2 = range(frame.shape[1])

    # Compute and plot left dendrogram #
    if row_method:
        d1 = dist.pdist(frame)
        D1 = dist.squareform(d1)
        ax1 = fig.add_axes([ax1_x, ax1_y, ax1_w, ax1_h], frame_on=True)
        Y1 = sch.linkage(D1, method=row_method, metric=row_metric)
        Z1 = sch.dendrogram(Y1, orientation='right')
        ind1 = sch.fcluster(Y1, 0.7*max(Y1[:,2]), 'distance')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ### apply the clustering for the array-dendrograms to the actual matrix data
        idx1 = Z1['leaves']
        frame = frame.iloc[idx1,:]
        ### reorder the flat cluster to match the order of the leaves the dendrogram
        ind1 = ind1[idx1]
    else: idx1 = range(frame.shape[0])

    # Plot distance matrix #
    axm = fig.add_axes([axm_x, axm_y, axm_w, axm_h])
    axm.matshow(frame, aspect='auto', origin='lower', cmap=cmap, norm=norm)
    axm.set_xticks([])
    axm.set_yticks([])

    # Add text #
    new_row_header = []
    new_column_header = []
    for i in range(frame.shape[0]):
        axm.text(frame.shape[1]-0.5, i, '  ' + row_header[idx1[i]], verticalalignment="center")
        new_row_header.append(row_header[idx1[i]] if row_method else row_header[i])
    for i in range(frame.shape[1]):
        axm.text(i, -0.9, ' '+column_header[idx2[i]], rotation=90, verticalalignment="top", horizontalalignment="center")
        new_column_header.append(column_header[idx2[i]] if column_method else column_header[i])

    # Plot column side colorbar #
    if column_method:
        axc = fig.add_axes([axc_x, axc_y, axc_w, axc_h])
        cmap_c = matplotlib.colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])
        dc = numpy.array(ind2, dtype=int)
        dc.shape = (1,len(ind2))
        axc.matshow(dc, aspect='auto', origin='lower', cmap=cmap_c)
        axc.set_xticks([])
        axc.set_yticks([])

    # Plot column side colorbar #
    if row_method:
        axr = fig.add_axes([axr_x, axr_y, axr_w, axr_h])
        dr = numpy.array(ind1, dtype=int)
        dr.shape = (len(ind1),1)
        cmap_r = matplotlib.colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])
        axr.matshow(dr, aspect='auto', origin='lower', cmap=cmap_r)
        axr.set_xticks([])
        axr.set_yticks([])

    # Plot color legend #
    ### axes for colorbar
    axcb = fig.add_axes([axcb_x, axcb_y, axcb_w, axcb_h], frame_on=False)
    cb = matplotlib.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, orientation='horizontal')
    axcb.set_title("colorkey")
    max_cb_ticks = 5
    axcb.xaxis.set_major_locator(pyplot.MaxNLocator(max_cb_ticks))

    # Render the graphic #
    if len(row_header)>50 or len(column_header)>50: pyplot.rcParams['font.size'] = 5
    else: pyplot.rcParams['font.size'] = 8

    # Return figure #
    return fig, axm, axcb, cb

###############################################################################
def plot_hierarchical_heatmap(data_matrix, row_names, column_names):
    df = pandas.DataFrame(data_matrix, index=row_names, columns=column_names)

    frame = df
    fig, axm, axcb, cb = plot_hierarchical_heatmap_core(df)
    cb.set_label("Correlation Coefficient")


# SC/intra/hierarchical: print + save networks
try:
  os.mkdir('plots')
except:
    pass
for i in range(4):
    # plot_hierarchical_heatmap(mat[i], seed_names, seed_names)
    # plt.savefig("plots/hierarchical_consensus_{}.png".format(i + 1), dpi=500)
    # plt.savefig("plots/hierarchical_consensus_{}.pdf".format(i + 1), dpi=500)
    # plt.close()

    set_text_size(8)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    max = np.abs(mat[i]).max()
    m = ax.matshow(mat[i], vmax=max, vmin=-max, cmap='RdBu_r')
    ax.set_xticks(np.arange(mat[i].shape[1]))
    ax.set_yticks(np.arange(mat[i].shape[0]))
    ax.set_xticklabels(seed_names, rotation='vertical')
    ax.set_yticklabels(seed_names)
    fig.colorbar(m)
    plt.savefig("plots/yeo7node_{}_2212.png".format(i + 1), dpi=300, tight_layout=True)


v_labels = dataset['MRIQ_labels']

#plt.show()
set_text_size(12)
fig = plt.figure(figsize=(3, 5))
ax = fig.add_subplot(111)
max = np.abs(v).max()
m = ax.matshow(v, vmax=max, vmin=-max, cmap='RdBu_r')
fig.colorbar(m)
plt.yticks(range(31), v_labels)
plt.xticks(range(4), range(1, 5))
plt.savefig('./plots/questions_2212.png', dpi=300, tight_layout=True)

