import numpy as np
import warnings

def quadratic1(a,b,c):
    '''
    Solve an quadratic equation
    '''
    if b**2-4*a*c < 0: x = np.nan
    elif b**2-4*a*c == 0: x = -b/(2*a)
    else: x = np.array(((-b+np.sqrt(b**2-4*a*c))/(2*a), (-b-np.sqrt(b**2-4*a*c))/(2*a)))
    return x

def demean(d):
    return d - d.mean(axis=0)[np.newaxis, :]


def is_outliers(data):
    '''
    Check outliers.
    Checking if the giving number's zscore is above 2.5.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(data, axis=0)
        sd = np.sqrt(np.nanmean((data - mean)**2, axis=0))
    zdata = (data - mean) / sd
    return abs(zdata) > 2.5

def imputedata(data, strategy='mean'):
    '''
    impute outliers and missing data
    impute outlier with mean or mean+-2sd
    missing data (np.nan) will impute as mean
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(data, axis=0)
        std = np.sqrt(np.nanmean((data - mean)**2, axis=0))

    data_sign = np.sign(data - mean)
    data_sign[np.isnan(data_sign)] = 0 # missing data will be imputed as mean

    is_out = is_outliers(data)
    data[is_out] = np.nan

    for i in range(data.shape[1]):
        ind_nan = np.where(np.isnan(data[:, i]))
        if strategy == '2sd':
            data[ind_nan, i] = mean[i] + (std[i] * 2 * data_sign[ind_nan, i])
        if strategy == 'mean':
            data[ind_nan, i] = mean[i]
    return data

def mean_nonzero(data, axis):
    '''
    calculate the non zero elements in a 2-D matrix
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        temp_sum = np.sum(data != 0, axis=axis)
        temp_sum[temp_sum == 0] = 1
        return np.sum(data, axis=axis)/temp_sum

import pickle
def load_pkl(file):
    '''
    load pickled file
    '''
    with open(file, 'rb') as handle:
        return pickle.load(handle)
        

def flatten(corrmat):
    '''
    flatten the correlation coefficient matrix in to a flat vector
    '''

    triu_inds = np.triu_indices(corrmat.shape[0], 1)
    corrmat_vect = corrmat[triu_inds]
    return corrmat_vect


def unflatten(corrmat_vect):
    '''
    Transform the flattened correlation matrice back to matrices
    for visualisation
    '''
    
    # figure out the size
    y = corrmat_vect.shape[0]
    x = quadratic1(0.5, -0.5, -y)
    x = int(np.max(x))
    
    idx = np.triu_indices(x, 1)
    corr_mat = np.zeros((x, x))
    corr_mat[idx] = corrmat_vect
    corr_mat = corr_mat + corr_mat.T
    
    return corr_mat
