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
sns.set_context('talk', font_scale=1)
f, ((ax11, ax12, ax13), (ax21, ax22, ax23) ) = plt.subplots(
                2, 3, figsize=(8, 6),
                sharey='row')

axes_lst = [ax11, ax21, ax12, ax22, ax13, ax23]

for s, ax, label in zip(sig_sets, axes_lst, labels):
    x = dataset[s[0]]
    y = dataset[s[1]]
    X = df_X.drop(s[0], axis=1)

    g = sns.regplot(x, y, x_partial=X, y_partial=X,
            marker='.', color='black', line_kws={"lw":1}, ax=ax)
    ax.set_ylim(-3, 3)

    sns.despine()

    ax.set_xlabel('{}'.format(label[0]))
    ax.set_ylabel('{}'.format(label[1]))
f.tight_layout()
# f.subplots_adjust(top=0.92)
# f.suptitle('Adjested Variable Plots', fontsize='x-large')
f.savefig('./reports/revision/plots/av-plots.png', dpi=300)
