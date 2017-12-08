'''
File input/output for this project ONLY
'''
import copy
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats.mstats import zscore

import joblib

from nilearn.signal import clean

def save_output(dataset, confound, best_clf, X, Y, path):
    '''
    save the scores and cognitve measures in pandas dataframe

    '''
    dict_info = {     
        'ID'              : dataset['IDs'], 
        'Age'             : list(dataset['Age'].flatten()), 
        'Gender'          : list(dataset['Gender'].flatten()), 
        'Motion_Jenkinson': list(dataset['Motion_Jenkinson'].flatten())
    }

    info = pd.DataFrame.from_dict(dict_info).set_index('ID')

    cogmeasure = pd.DataFrame(data=dataset['CognitiveMeasures'], 
                              index=dataset['IDs'], 
                              columns=dataset['CognitiveMeasures_labels'])

    df = pd.concat([info, cogmeasure], axis=1)
    
    # regress out confound
    z_confound = zscore(confound)
    # squared measures to help account for potentially nonlinear effects of these confounds
    z2_confound = z_confound ** 2
    conf_mat = np.hstack((z_confound, z2_confound))

    # clean signal
    z_data = clean(zscore(df), confounds=conf_mat, detrend=False, standardize=False)
    df_z = pd.DataFrame(data=z_data, index=dataset['IDs'], columns=df.columns)
    
    # save the scca scores
    X_scores, Y_scores = best_clf.transform(X, Y)

    z_cc = (zscore(X_scores) + zscore(Y_scores)) / 2

    # flip component 2 here 
    cca = pd.DataFrame(data=z_cc,
                      index=dataset['IDs'],
                      columns=['CC_{:02d}'.format(i + 1) for i in range(z_cc.shape[1])])

    df_z = df = pd.concat([df_z, cca], axis=1)
    return X_scores, Y_scores, df_z
