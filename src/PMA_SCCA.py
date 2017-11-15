'''Sparse Canonical Correlation Analysis from R package 'PMA'.

Talk to Berin about cleaning this shit up.

'''
# H. Wang
# Install R package 'PMA' and dependencies before using this module.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.stats.mstats import zscore

from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from sklearn.linear_model import LinearRegression


# class _CCABase(object):
#     def __init__(self, penalty=None, n_component=None, verbose=False):
#         self.penalty = None
#         self.n_component = None
#         self.verbose = False


class SCCA(object):
    '''
    Sparse Canonical Correlation Analysis from R package 'PMA'.

    ref: http://cran.r-project.org/web/packages/PMA/PMA.pdf

    Parameters
    ----------
    penX, penY : float
        penalty on X and Y, between 0 and 1.
        Default as 1 (meaning no penalty)

    numSCC : int
        number of canonical components to keep.
        the default value is the maxium is the number of variables of the smaller dataset between X and Y.


    Example
    ----------
    >>> from PMA_SCCA import SCCA
    >>> model = SCCA(penX=0.6, penY=0.6)
    >>> model.fit(X, Y)
    >>> model.summary()
    >>> u, v = model.u, model.v


    '''
    def __init__(self, n_components=None, n_iter=50, scale=True ,penX=None, penY=None):
        penX = 1 if penX is None else penX
        penY = 1 if penY is None else penY

        self.penX = penX
        self.penY = penY

        self.n_components = n_components
        self.n_iter = n_iter
        self.scale = scale

        self.u = None
        self.v = None

        self.component_evs_summary = None
        self.component_evs_plot = None

        self.model_evs_summary = None
        self.model_evs_plot = None

    def fit(self, X, Y):
        '''
        Run SCCA on data set X and Y.

        ref: http://cran.r-project.org/web/packages/PMA/PMA.pdf

        Parameters
        ----------
        X, Y : array
            Your data here. Normalise the data yourself.
            The order of observation should match between X and Y
            Trouble shooting TBC

        Return
        ----------
        self.cancorrs : array, float
            correlations of the canonical components on the training dataset

        self.u, self.v : array, float
            canonical components of X and Y

        self.scores: tuple
            canonical component scores of X and Y

        self.ev: tuple
            explained variance of the given model fitting X and Y
        '''
        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        X, Y, self.x_mean, self.y_mean, self.x_std, self.y_std = _center_xy(X, Y, scale=self.scale)

        numpy2ri.activate()
        rPMA = importr('PMA') # import the R package
        # calculate the output.
        out = rPMA.CCA(x=X, z=Y, K=self.n_components, \
                niter=self.n_iter, standardize=False, \
                typex='standard', typez='standard', \
                penaltyx=self.penX, penaltyz=self.penY, \
                trace=False)
        # convert outputs back to Python objects
        # the bridge function in python use 0-index now (R is 1-index)
        self.u = numpy2ri.ri2py(out[0])
        self.v = numpy2ri.ri2py(out[1])
        numpy2ri.deactivate()
        self.x_score_, self.y_score_ = self.transform(X, Y)
        self.cancorr_= _cancorr(X, Y, self.u, self.v)
        return self

    def transform(self, X, Y):
        '''
        calculate the canonical scores of the current model with some given scores
        '''
        if self.scale:
            Xk, Yk = zscore(X), zscore(Y)
        else:
            Xk, Yk = X, Y
        return Xk.dot(self.u), Yk.dot(self.v)

    def score(self, X, Y):
        '''
        Returns the coefficient of determination R^2
        '''
        if self.scale:
            Xk, Yk = zscore(X), zscore(Y)
        else:
            Xk, Yk = X, Y
        cancorr = _cancorr(Xk, Yk, self.u, self.v)
        r2_sum = (cancorr ** 2).sum()
        return r2_sum

    # def predict(self, X):
    #     X = zscore(X)
    #     Ypred = np.dot(X, self.coef_.T)
    #     return Ypred

    # def summary(self):
    #     '''

    #     print the summary table of SCCA

    #     '''
    #     print 'Penalty - X: {}; Penalty - Y: {}'.format(self.penX, self.penY)
    #     print 'Size of the canonical weights of X:', self.u.shape
    #     print 'Size of the canonical weights of Y:', self.v.shape

    def component_explained(self, X, Y, show_results=True):
        '''
        calculate the explained variable percentage of all possible components.
        '''
        def plot_expvar(self, ev_componnets_summary):
            '''
            PLOT THE RESULTS

            clean up this bit
            '''
            fig = plt.figure()
            for i in range(2):

                if i == 0:
                    cur = 'X'
                else:
                    cur = 'Y'

                plt.subplot(211 + i)
                plt.plot(np.arange(self.n_components) + 1, ev_componnets_summary[i, :], label='exp var')
                plt.ylim(0, ev_componnets_summary[i, :].max())
                plt.ylabel('Explained Variance %')
                plt.xlim(1, self.n_components)
                plt.xlabel('Component Number')
                plt.xticks(range(0, self.n_components + 1)[1:])
                plt.title('{} total % variance explained'.format(cur))
            return fig
        # save the original number of components set by the user
        original_n = self.n_components

        # obtain the maximum number of components
        self.n_component = np.min((X.shape[1], Y.shape[1]))

        if self.scale:
    	    Xk, Yk = zscore(X), zscore(Y)
        else:
	        Xk, Yk = X, Y
        # fit data
        SCCA.fit(self, Xk, Yk)
        u, v = self.u, self.v

        # explained variance of each component
        x_ev_componnets = []
        y_ev_componnets = []

        for i in range(self.n_components):
            x_ev_c = _Rsquare(Xk, u[:, i:(i + 1)])
            y_ev_c = _Rsquare(Yk, v[:, i:(i + 1)])
            x_ev_componnets.append(x_ev_c)
            y_ev_componnets.append(y_ev_c)

        ev_componnets_summary = np.vstack((np.array(x_ev_componnets), np.array(y_ev_componnets)))

        self.component_evs_summary = np.transpose(
            np.array([range(1, self.n_components + 1)] + [x_ev_componnets] + [y_ev_componnets]))
        self.component_evs_plot = plot_expvar(self, ev_componnets_summary)

        self.n_components = original_n

        if show_results:
            table_expvar(self.component_evs_summary)
            self.component_evs_plot.show()

        return self

    def model_explained(self, X, Y, show_results=True):
        '''
        calculate the explained variable percentage by numbers of componets in a model.
        '''

        def plot_expvar(exp_var_X, exp_var_Y, limit_exp_var):
            '''
            PLOT THE RESULTS
            '''
            fig = plt.figure()
            plt.plot(np.arange(limit_exp_var) + 1, exp_var_X, label='X model exp var')
            plt.plot(np.arange(limit_exp_var) + 1, exp_var_Y, label='Y model exp var')
            plt.ylim(-0.1, 1)
            plt.ylabel('Explained Variance %')
            plt.xlim(1, limit_exp_var)
            plt.xlabel('Number of components')
            plt.xticks(range(0, limit_exp_var + 1)[1:])
            plt.legend()
            return fig

        limit_exp_var = np.min([X.shape[1], Y.shape[1]]) #save for later
        limit_exp_var = np.asscalar(limit_exp_var)

        exp_var_X = []
        exp_var_Y = []

        original_n = self.n_components

        if self.scale:
        	Xk, Yk = zscore(X), zscore(Y)
        else:
	        Xk, Yk = X, Y

        for i in range(1, limit_exp_var + 1):
            self.n_components = i
            SCCA.fit(self, Xk, Yk)
            x_ev_c = _Rsquare(Xk, self.u)
            y_ev_c = _Rsquare(Yk, self.v)
            exp_var_X.append(x_ev_c)
            exp_var_Y.append(y_ev_c)

        self.n_components = original_n

        self.model_evs_summary = np.transpose(
            np.array([range(1, limit_exp_var + 1)] + [exp_var_X] + [exp_var_Y]))
        self.model_evs_plot = plot_expvar(exp_var_X, exp_var_Y, limit_exp_var)

        if show_results:
            table_expvar(self.model_evs_summary)
            self.model_evs_plot.show()

        return self


def _Rsquare(X, P):
    '''
    calculate the coefficent of determination (R square):
    the ratio of the explained variation to the total variation.
    '''
    lr = LinearRegression(fit_intercept=False)
    lr.fit(P, X.T)
    rec_ = lr.coef_.dot(P.T)
    return 1 - (np.var(X - rec_) / np.var(X))

def _cancorr(X, Y, u, v):
    '''
    Calculate the canonical correlation
    '''
    n_components = u.shape[1]
    x_score = X.dot(u)
    y_score = Y.dot(v)
    cancorr = np.corrcoef(x_score.T, y_score.T)\
		    [n_components:, 0:n_components].diagonal()
    return cancorr

def table_expvar(ev_summary):
    '''
    summary the explained variable tabel
    '''
    np.set_printoptions(precision=3, suppress=True, linewidth=1000)
    print ''
    print 'Explained variance %'
    print '    n', '  exp_brain', '  exp_behaviour'
    print ev_summary


# def _scca(x, y, v, typex, typez, penaltyx, penaltyz, niter, trace, upos, uneg, vpos, vneg,chromx,chromz):
#     _v = np.random.normal(0, 1, v.size)
#     u = np.random.normal(0, 1, x.shape[1])
#     for i in range(niter):
#         if np.sum(np.isnan(u)) > 0 and np.sum(np.isnan(v)) > 0:
#             v = np.zero(v.size)
#             _v = v
#         if  np.sum(np.abs(_v - v)) > 1e-6:
#             if trace: print(i,)
#             u_cur = np.empty(x.shape[1])
#             if typex is 'standard':
#                 argu <- y.dot(v).dot(x)
#                 if upos: argu = np.maximum(argu, 0)
#                 if uneg: argu = np.minimum(argu, 0)
#                 lamu =
#     return None

def _center_xy(X, Y, scale=True):
    x_mean = np.mean(X, axis=0)
    X -= x_mean
    y_mean = np.mean(Y, axis=0)
    Y -= y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std
