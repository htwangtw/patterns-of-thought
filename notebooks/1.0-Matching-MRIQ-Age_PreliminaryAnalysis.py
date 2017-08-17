
# coding: utf-8

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_MRIQ = pd.read_csv('./RawData/Assessment_raw/8100_MRIQ_20161025.csv', header=0, index_col=None)
data_MRIQ_copy = data_MRIQ.copy()

# loading the age data
data_age = pd.read_csv('./RawData/Assessment_raw/8100_Age_20161025.csv', header=1, index_col=None)

data_MRIQ = data_MRIQ.sort_values(by=['Anonymized ID', 'Visit'])

#checking missing data situation
data_MRIQ = data_MRIQ_copy.dropna(thresh=20)
print '{:>55}'.format('Number of observations'),':','{:>5}'.format(data_MRIQ_copy.shape[0])
print '{:>55}'.format('Number of observations with less than 20 missing data'),':','{:>5}'.format(data_MRIQ.shape[0])


def variable_summary(df, label):
    variable = df[label].values.tolist()
    count_v = [variable.count(x) for x in variable]
    dic_v = dict(zip(variable, count_v))
    print '\nNumber of subjects under each %s\n' %label
    for t, v in sorted(dic_v.iteritems()):
        print '{:>25}'.format(t),':','{:>5}'.format(v)

variable_summary(data_MRIQ, 'Subject Type')
variable_summary(data_MRIQ, 'Visit')
variable_summary(data_age, 'Visit')


# merge the Age and MRIQ data
selected_age = []
for index, row in data_MRIQ.iterrows():
    tmp_id = row['Anonymized ID']
    tmp_visit = row['Visit']

    # in the MRIQ data replace V4R as V4
    #                          V2R as V2REP
    #                          V1 as V2
    if tmp_visit == 'V1':
        tmp_visit = 'V2'
    if tmp_visit == 'V4R':
        tmp_visit = 'V4'
    if tmp_visit == 'V2R':
        tmp_visit = 'V2REP'

    #find the id in data_age
    idx_age = data_age[data_age['Anonymized ID'] == tmp_id].index.tolist()

    if len(idx_age) == 1: # one visit
        tmp_age = float(data_age.loc[idx_age]['AGE_04'].values)
    elif len(idx_age) > 1: # multiple visits
        tmp_age_cases = data_age.loc[idx_age]
        try:
            tmp_age = float(tmp_age_cases[tmp_age_cases['Visit'] == tmp_visit]['AGE_04'].values)
        except TypeError:
            tmp_age = float(tmp_age_cases[tmp_age_cases['Visit'] == tmp_visit[:2]]['AGE_04'].values)
    else: # no age data
        tmp_age = np.nan
    selected_age.append(tmp_age)

data_MRIQ['AGE'] = selected_age


#save the data_MRIQ as csv, go to SPSS to do PCA
data_MRIQ.sort_values(by=['AGE']).to_csv('./data/Behavioural/NKI_MRIQ_Age_merged.csv', sep=',', index=False)


'''Age information'''


hist_age = data_age.sort_values(by=['AGE_04']).dropna(how='any').values[:, -1]
n, bins, patches = plt.hist(hist_age, 25, facecolor='green')
plt.xlabel('Age')
plt.ylabel('Number of subjects')
plt.title('NKI data: age distribution (N=%i) %i')
plt.show()

hist_MRIQ_age = data_MRIQ.sort_values(by=['AGE']).dropna(how='any').values[:, -1]
n, bins, patches = plt.hist(hist_MRIQ_age, 25, facecolor='green')
plt.xlabel('Age')
plt.ylabel('Number of subjects')
plt.title('NKI data: age distribution of data set MRIQ')
plt.show()
print 'The number of cases between Age 18 - 55: ', \
        (data_MRIQ['AGE'].values <= 55).sum() - (data_MRIQ['AGE'].values < 18).sum()
