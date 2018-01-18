import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats

df_path = "./data/processed/NYCQ_CCA_score_revision_yeo7nodes_4_0.8_0.5.pkl"
dataset = pd.read_pickle(df_path)
df_X = dataset.loc[:, 'CC_01':'CC_04']
df_Y = dataset.loc[:, 'DKEFSCWI_40':'WIAT_08']

sig_sets = [
        ('CC_01', 'PROV_16'),
        ('CC_03', 'PROV_16'),
        ('CC_01', 'INT_17'),
        ('CC_03', 'INT_17'),
        ('CC_01', 'WIAT_08'),
        ('CC_03', 'WIAT_08'),
]


def r_pearson(x, y):
    return stats.pearsonr(x, y)[0]


for s in sig_sets:
    x = dataset[s[0]]
    y = dataset[s[1]]
    X = df_X.drop(s[0], axis=1)
    Y = df_Y.drop(s[1], axis=1)
    dat = pd.concat((X, Y), axis=1)
    if s[0] == 'CC_03':
        x = - dataset[s[0]]
    lr = LinearRegression(fit_intercept=True)
    lr.fit(dat, x)
    x_res = x - lr.predict(dat)
    lr = LinearRegression(fit_intercept=True)
    lr.fit(dat, y)
    y_res = y - lr.predict(dat)

    plt.close("all")
    fig = sns.jointplot(x, y, kind="reg", stat_func=r_pearson)
    fig.savefig('./reports/{}_{}.png'.format(s[0], s[1]))

    plt.close("all")
    fig = sns.jointplot(x_res, y_res, kind="reg", stat_func=r_pearson)
    fig.savefig('./reports/{}_{}_res.png'.format(s[0], s[1]))
