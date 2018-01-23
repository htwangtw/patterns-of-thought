import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats

df_path = "./data/processed/NYCQ_CCA_score_revision_yeo7nodes_4_0.8_0.5.pkl"
dataset = pd.read_pickle(df_path)
dataset.loc[:, 'CC_03'] = - dataset.loc[:, 'CC_03']
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


for s in sig_sets:
    x = dataset[s[0]]
    y = dataset[s[1]]
    X = df_X.drop(s[0], axis=1)

    plt.close("all")
    sns.set_style({"font.sans-serif": ["Arial"]})
    f, ax = plt.subplots(figsize=(5, 6))
    g = sns.regplot(x, y, x_partial=X, y_partial=X, ax=ax)
    g.set(ylim=(-3, 3))
    sns.despine()
    f.savefig('./reports/plots/resplots/{}_{}.png'.format(s[0], s[1]))
