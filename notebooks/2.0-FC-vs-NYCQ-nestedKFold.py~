# coding: utf-8
import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

import joblib
from sklearn.model_selection import KFold, ParameterGrid
# load my modules
from src.file_io import save_output
from src.PMA_SCCA import SCCA
from src.utils import load_pkl

dat_path = './data/processed/dict_SCCA_data_prepro_11082017.pkl'
# load data
dataset = load_pkl(dat_path)
X = dataset['FC_Yeo7']
Y   = dataset['MRIQ']

'''
K-Fold CV for parameters/model selection

The objective metirce is cumulative R^2,
It's calculated by the sum of squred canonical correlations in the model,
It represent the amount of variance in one data set explained by the other dataset’s variate.
High explained variance suggests a high ability to predict between the two data set.

The problem of this metirce is that
model will always perform better with higher number of components,
and in the case of high component number, the sparsity reduces.
Therefore I decided the number of components in the model 
before searching for the best parameter.
'''
## Decide factor number
model = SCCA(n_components=None, scale=True,
                n_iter=50, penX=1, penY=1)
model.model_explained(X, Y)
model.model_evs_plot
plt.savefig('./reports/gridsearch_correct/select_n_comp.png')

## Penalty selection CV
# Grid search range
param_grid = {
    'n_factor':np.array([4]),
    'reg_X': np.array(range(1, 10)) * 0.1,
    'reg_Y': np.array(range(1, 10)) * 0.1,}

## deciding the fold
# the populaiton sample size is 256,
# test set ~= 50
# training set ~=160
# validation set ~=40
out_folds = 5
in_folds = 5

# set up inner and outer folds and the parameter grid
KF_out = KFold(n_splits=out_folds, shuffle=True, random_state=1)
KF_in = KFold(n_splits=in_folds, shuffle=True, random_state=1)
para_sets = ParameterGrid(param_grid)

# save the training results of the inner fold
train_mean_evs = np.zeros((out_folds, len(para_sets)))
train_sem_evs  = np.zeros((out_folds, len(para_sets)))
# save the testing results from the outer fold
test_evs = np.zeros(out_folds)
# initialise.....
best_model = None
best_ev  = 0

# train-test split
for i, (train_index, test_index) in enumerate(KF_out.split(X, Y)):
    X_train_outer, y_train_outer = X[train_index, :], Y[train_index, :]
    X_test_outer, y_test_outer = X[test_index, :], Y[test_index, :]
    
    print('Start fold {:01d}'.format(i + 1))
    
    out_best_model = None
    out_best_ev  = 0
    
    for j, parameters in enumerate(iter(para_sets)):   
        train_ev_list = []
        lat_best_ev   = 0
        lat_best_model  = None
        # learning parameters
        model = SCCA(n_components=parameters['n_factor'], scale=True, n_iter=50,
                     penX=parameters['reg_X'], penY=parameters['reg_Y'],
                    )
        
        for train_index, test_index in KF_in.split(X_train_outer, y_train_outer):
            # gather train-test data
            X_train_inner, Y_train_inner = X_train_outer[train_index, :], y_train_outer[train_index, :]
            X_test_inner, Y_test_inner = X_train_outer[test_index, :], y_train_outer[test_index, :]
            
            # fit the training set
            model.fit(X_train_inner, Y_train_inner)
            # calculate sum of R^2 on the validation set
            # model.score returns canonical correlations
            r2_train = model.score(X_test_inner, Y_test_inner)
            
            # dump to list
            train_ev_list.append(r2_train)

            if r2_train > lat_best_ev:
                # select the best model of the current parameter set
                lat_best_model = copy.deepcopy(model)
                lat_best_ev = r2_train 
        
        # Calculate the mean and sem, dump
        train_mean_evs[i, j] = np.mean(train_ev_list)
        train_sem_evs[i, j] = sem(train_ev_list, ddof=1)

        if lat_best_ev > out_best_ev:
            # select the best model of the parameter set 
            out_best_ev = lat_best_ev
            out_best_model = copy.deepcopy(lat_best_model)
        
    # train on the whole training srt and test set and save the cumulative R^2
    out_best_model.fit(X_train_outer, y_train_outer)
    r2_test = model.score(X_test_outer, y_test_outer)
    # dump
    test_evs[i] = r2_test
    
    # save the model of the best model of this fold
    joblib.dump(out_best_model, 
            './models/SCCA_fold{:1d}_{:1d}_{:.2f}_{:.2f}.pkl'.format(
                i + 1, out_best_model.n_components, out_best_model.penX, out_best_model.penY)) 
    
    print('==================================================')
    if r2_test > best_ev:
        # select the best model across the outer folds
        print('\nNew Best model: \n {:} components,penalty x: {:}, penalty y: {:}\nOOS performance: {}'.format(
            out_best_model.n_components, out_best_model.penX,  out_best_model.penY,  r2_test))   
        best_model = copy.deepcopy(out_best_model)
        best_ev = r2_test
    print('\n==================================================')
    
print('\nNew Best model: \n {:} components,penalty x: {:}, penalty y: {:}\nO-O-S performance: {}'.format(
            best_model.n_components, best_model.penX, best_model.penY, best_ev))
# save the best model
joblib.dump(best_model, 
            './models/SCCA_best_{:1d}_{:.2f}_{:.2f}.pkl'.format(
                best_model.n_components, best_model.penX, best_model.penY)) 

# plotting the performance of each fold 
labels = []
for par in iter(para_sets):
    labels.append('{:} x {:}'.format(par['reg_X'], par['reg_Y']))

plt.close('all')
set_text_size(11)
plt.figure(figsize=(15, 6))
for i in range(test_evs.shape[0]):
    plt.plot(range(0 ,len(labels)) , train_mean_evs[i, :], 
         color='b', alpha=0.3)
    
plt.plot(range(0 ,len(labels)) , train_mean_evs.mean(axis=0), 
         color='r', label="Mean performance across folds")
plt.fill_between(range(0 ,len(labels)), 
                 train_mean_evs.mean(axis=0) + train_sem_evs.mean(axis=0), 
                 train_mean_evs.mean(axis=0) - train_sem_evs.mean(axis=0), 
                 facecolor='grey', alpha=0.3, label="Standard Error of the Estimate")
plt.legend()
plt.ylim(0, 0.3)
plt.xlabel('L1 Penalty: X x Y')
plt.xticks(range(0 ,len(labels)), labels, rotation=90)
plt.ylabel('Cumulative Explained Variance')
plt.savefig('./reports/ev_per_para.png', bbox_inches='tight')
