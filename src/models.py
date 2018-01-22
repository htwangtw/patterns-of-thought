import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.stats.mstats import zscore
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

from nilearn.signal import clean

from src.PMA_SCCA import SCCA
from src.visualise import set_text_size

def clean_confound(RS, COG, confmat):
    '''
    We first created the confound matrix according to Smith et al. (2015).
    The confound variables are motion (Jenkinson), sex, and age.
    We also created squared confound measures to help account for potentially nonlinear effects of these confounds.
    '''

    # regress out confound
    z_confound = zscore(confmat)
    # squared measures to help account for potentially nonlinear effects of these confounds
    z2_confound = z_confound ** 2
    conf_mat = np.hstack((z_confound, z2_confound))

    # clean signal
    RS_clean = clean(np.arctanh(RS), confounds=conf_mat, detrend=False, standardize=False)
    COG_clean = clean(zscore(COG), confounds=conf_mat, detrend=False, standardize=False)

    return RS_clean, COG_clean, conf_mat

def search_grid_scca(reg_X, reg_Y):
    '''
    reg_X, reg_Y: tuple
    (lower limit, upper limit)

    n_selected: list
    list of component number
    '''

    param_setting = {
        'reg_X': np.arange(reg_X[0], reg_X[1] + 0.1, 0.1),
        'reg_Y': np.arange(reg_Y[0], reg_Y[1] + 0.1, 0.1),}
    param_grid = ParameterGrid(param_setting)
    return param_grid

def nested_kfold_cv_scca(X, Y, R=None, n_selected=4, out_folds=5, in_folds=5, reg_X=(0.1, 1), reg_Y=(0.1, 1)):
    '''
    TBC
    '''
    grid = search_grid_scca(reg_X, reg_Y)
    KF_out = KFold(n_splits=out_folds, shuffle=True, random_state=1)
    KF_in = KFold(n_splits=in_folds, shuffle=True, random_state=1)

    n_penX = int(reg_X[1] * 10 - reg_X[0] + 1)
    n_penY = int(reg_Y[1] * 10 - reg_Y[0] + 1)

    best_model = None
    best_score = 0
    pred_scores = []
    para_search = np.zeros((n_penX, n_penY, out_folds))
    for i, (train_idx, test_idx) in enumerate(KF_out.split(X)):
        print('Fold {0:}/{1:}'.format(i + 1, out_folds))
        X_discovery, X_test = X[train_idx], X[test_idx]
        Y_discovery, Y_test = Y[train_idx], Y[test_idx]
        if R is not None:
            R_discovery, R_test = R[train_idx], R[test_idx]

        para_mean_score = np.zeros((n_penX, n_penY))
        for j, parameters in enumerate(iter(grid)):
            para_idx = np.unravel_index(j, para_mean_score.shape) # (C_x,C_y)
            model = SCCA(n_components=n_selected, scale=True, n_iter=100,
                         penX=parameters['reg_X'], penY=parameters['reg_Y'],
                        )
            inner_scores = []
            for k, (train_idx, test_idx) in enumerate(KF_in.split(X_discovery)):
                # find best weights for this hyper parameter set
                X_train, X_confirm = X_discovery[train_idx], X_discovery[test_idx]
                Y_train, Y_confirm = Y_discovery[train_idx], Y_discovery[test_idx]
                if R is not None:
                    R_train, R_confirm = R_discovery[train_idx], R_discovery[test_idx]
                    X_train, Y_train, R_train = clean_confound(X_train, Y_train, R_train)
                    X_confirm, Y_confirm, R_confirm = clean_confound(X_confirm, Y_confirm, R_confirm)

                model.fit(X_train, Y_train)

                pred_ev = model.score(X_confirm, Y_confirm)

                inner_scores.append(pred_ev)

            para_mean_score[para_idx] = np.mean(inner_scores)

        idx = np.argmax(para_mean_score)
        d_idx = np.unravel_index(idx, para_mean_score.shape)
        C = 0.1 * (np.array(d_idx) + 1)

        if R is not None:
            X_discovery, Y_discovery, R_discovery = clean_confound(X_discovery, Y_discovery, R_discovery)
            X_test, Y_test, R_test = clean_confound(X_test, Y_test, R_test)

        para_best_model = SCCA(n_components=n_selected, scale=True, n_iter=100,
                         penX=C[0], penY=C[1],
                        )
        para_best_model.fit(X_discovery, Y_discovery)
        pred_ev = para_best_model.score(X_test, Y_test)
        pred_scores.append(pred_ev)
        parameter_grid(para_mean_score, pred_ev, i)
        plt.show()
        if pred_ev > best_score:
            best_score = pred_ev
            best_model = copy.deepcopy(para_best_model)
            print('\nNew Best model: \n {:} components,penalty x: {:}, penalty y: {:}\nOOS performance: {}'.format(
            best_model.n_components, best_model.penX,  best_model.penY,  best_score))
        para_search[..., i] = para_mean_score

    # final parameter
    print('\nBest parameters based on outer fold ev results: X-{:}; Y-{:}\n'.format(
            best_model.penX, best_model.penY))

    return (para_search, best_model, pred_scores)

def parameter_grid(para, pred_evs, i):
    '''
    plot the output of the grid search
    '''
    set_text_size(12)

    idx = np.argmax(para)
    d_idx = np.unravel_index(idx, para.shape)

    title = 'Sparsity Search - Fold {}'.format(i + 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    hm = ax.matshow(para.T, vmin=0, vmax=para.max(), cmap="inferno")
    ax.set_xticklabels(np.array(range(0, 10)) * 0.1)
    ax.set_yticklabels(np.array(range(0, 10)) * 0.1)
    ax.set_xlabel('Connectivity')
    ax.set_ylabel('MRIQ')
    ax.set_title(title)
    ax.xaxis.set_ticks_position('bottom')
    fig.colorbar(hm, label='CV Estimate of Prediction EV')
    # Create a Rectangle patch
    rect = patches.Rectangle(np.array(d_idx) - 0.5, 1, 1,linewidth=2,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    # add prediction error
    ax.annotate('Test EV:{:.3f}'.format(pred_evs), (0,0), (0, -40),
                     xycoords='axes fraction', textcoords='offset points', va='top',
                     fontstyle='italic', fontsize=10)
    return fig

def permutate_scca(X, Y, d, model, n_permute=1000, aug=True):
    '''
    find the best variates among all components
    by calculating the FWE-corrected p-value based on FDR
    '''
    X_orig, Y_orig = X, Y
    n_mod_select = model.n_components
    permute_cancorr = np.zeros((n_permute, n_mod_select))
    np.random.seed(42)
    for i in range(n_permute):
        if aug:
            aug_idx = np.random.randint(1, Y_orig.shape[0], 1000)
            X, Y = X_orig[aug_idx], Y_orig[aug_idx]

        # permute the cognitive measures
        per_idx = np.random.permutation(Y.shape[0])
        cur_y = Y[per_idx, :]

        permute_model = copy.deepcopy(model)
        permute_model.fit(X, cur_y)
        cur_cancorr = permute_model.cancorr_
        permute_cancorr[i, :] = cur_cancorr

    # calculate the FWE-corrected p value
    p_val = (1 + np.sum(d < np.repeat(permute_cancorr[1:, 0:1], n_mod_select, axis=1), 0)) / float(n_permute)
    results = {
        'Component': range(1, n_mod_select + 1),
        'P-values': p_val,
        'alpah0.05': p_val < 0.05,
        'alpah0.01': p_val < 0.01,
        'alpah0.001': p_val < 0.001,
    }

    df_permute = pd.DataFrame.from_dict(results).set_index('Component')
    return df_permute
