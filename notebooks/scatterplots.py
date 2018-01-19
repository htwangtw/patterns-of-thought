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

labels = [
        ('Component 1', 'Proverb'),
        ('Component 3', 'Proverb'),
        ('Component 1', 'WASI'),
        ('Component 3', 'WASI'),
        ('Component 1', 'WIAT'),
        ('Component 3', 'WIAT')
        ]

plt.close("all")

sns.set_style({"font.sans-serif": ["Arial"]})
sns.set_context('talk', font_scale=1.2)
f, ((ax11, ax12),
        (ax21, ax22),
        (ax31, ax32)) = plt.subplots(
                3, 2, figsize=(6, 8),
                sharex='col', sharey='row')

axes_lst = [ax11, ax12, ax21, ax22, ax31, ax32]

for s, ax, label in zip(sig_sets, axes_lst, labels):
    x = dataset[s[0]]
    y = dataset[s[1]]
    X = df_X.drop(s[0], axis=1)

    g = sns.regplot(x, y, x_partial=X, y_partial=X,
            marker='.', color='black', ax=ax)
    ax.set_ylim(-3, 3)
    sns.despine()
    if label[1] == 'WIAT':
        ax.set_xlabel(label[0])
    else:
        ax.set_xlabel('')
    if label[0] == 'Component 1':
        ax.set_ylabel('{} | Others'.format(label[1]))
    else:
        ax.set_ylabel(' ')
f.tight_layout()
f.subplots_adjust(top=0.92)
f.suptitle('Adjested Variable Plots', fontsize='x-large')
f.savefig('./reports/plots/av-plots.png', dpi=300)
