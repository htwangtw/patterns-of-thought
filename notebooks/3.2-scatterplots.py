import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_path = "./data/processed/NYCQ_CCA_score_revision_withPC.csv"
dataset = pd.read_csv(df_path)
df_X = dataset.loc[:, 'CC_01':'CC_04']
Y = dataset.loc[:, 'task.pc'] * -1

plt.close("all")

sns.set_style({"font.sans-serif": ["Arial"]})
sns.set_context('talk', font_scale=1)
f, (ax1, ax2, ax3) = plt.subplots(
                1, 3, figsize=(7, 3))

df_PC = pd.DataFrame([-0.25213599, -0.04582192, -0.31767920, -0.29617578, -0.41599742, -0.15568316, -0.52272943, -0.52454787],
index=['CWI', 'TOWER', 'TMT', 'DF', 'PROV', 'VF', 'WASI', 'WIAT'])

sig_sets = ['CC_01', 'CC_03', 'CC_04']
axes_lst = (ax1, ax2, ax3)
for s, ax in zip(sig_sets, axes_lst):
    x = df_X[s]
    X = df_X.drop(s, axis=1)
    y = Y.values
    g = sns.regplot(x, y, x_partial=X, y_partial=X,
            marker='.', color='black', line_kws={"lw":1}, ax=ax)

    sns.despine()

    ax.set_xlabel('Component {}'.format(s[-1]))
    ax.set_ylabel('PC-Tasks')
f.tight_layout()
f.savefig('./reports/plots/av-plots_sigpc.png', dpi=300)

plt.close("all")
f = plt.figure(figsize=(2.5, 3))
ax = f.add_subplot(111)
m = ax.matshow(df_PC.values[:, 0:1] * -1,
                vmin=-0.6, vmax=0.6, cmap="RdBu_r")
ax.set_yticks(range(8))
ax.set_xticks(range(0))
ax.set_yticklabels(df_PC.index)
f.colorbar(m)
f.tight_layout()
f.savefig('./reports/plots/pc_tasks.png', dpi=300)
