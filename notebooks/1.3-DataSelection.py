
# coding: utf-8

# # Functional connectivity 
# Read data path and define result path, Create a list of atlas names, Extract time series and compute network connectivities per particiapnt. 
# 
# All done on YNiC server.
# pandas version >= 0.17.0
# # Matching behavioural and FC data

import csv
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

from src.utils import imputedata

'''
load data
'''

FC_subj = np.load('./data/interim/data_cross_corr_Yeo17_preprocessed_pptID.npy')
PATHS_FC = glob.glob('./data/interim/data_*_preprocessed.npy')
PATHS_ROI = glob.glob('./data/interim/data_*_ROI.npy')


def load_csv_pd(path, header_row_n):
    df = pd.read_csv(path, header=header_row_n ,na_values= ' ')
    return df.sort_values(by=['Anonymized ID'])

def select_data(df):
    prev_subj = None
    df['include'] = 0
    for idx, row in df.iterrows():
        cur_subj = row['Anonymized ID']
        if cur_subj != prev_subj and row['Anonymized ID'] in FC_subj:
            df.set_value(idx, 'include', 1)
        elif cur_subj == prev_subj and row['Anonymized ID'] in FC_subj:
            pass
        else:
            pass
        prev_subj = cur_subj
    return df.query('include == 1')


# get Assesment csv path
PATHS = ['./data/interim//NKI_MRIQ_Age_merged.csv', './data/interim/NKI_MotionParameters.csv'] + \
sorted(glob.glob('./data/raw/CognitiveTasks/*.csv')) + \
['./data/raw/Questionnaires/8100_BDI-II_20161025.csv',
    './data/raw/Questionnaires/8100_STAI_20161025.csv',
    './data/raw/Questionnaires/8100_UPPS-P_20161025.csv',
    './data/raw/Questionnaires/8100_Demos_20161025.csv'] 

# load variable name txt
task_names = []
var_names = []
with open('./data/raw/selected_CognitiveTasks_labels.txt', 'rb') as f:
    for line in f:
        task_names.append(line.split()[0])
        var_names.append(line.split()[1])
'''
filter data by missing meaures
'''

frame_df = []
print 'Number of participants with RS scan and task'
for i, path in enumerate(PATHS):
    
    task_name = path.split('/')[-1].split('.csv')[0].split('_')[1]
    
    if task_name in task_names:
        # for anything other than MRIQ and Motion
        df = load_csv_pd(path, header_row_n=1)
        df = select_data(df) # select participant
        df = df.set_index('Anonymized ID') # set ID as index for concatenation
        if task_name == 'Demos':
            df = df[~df.index.duplicated(keep='first')]
        # select the varables to save
        var_name = [var_names[i] for i, name in enumerate(task_names) if name == task_name]
        df = df[var_name]
        df = df.apply(pd.to_numeric, errors='coerce') # convert data to numerical values; if error, return nan
        frame_df.append(df)

    else:
        df = load_csv_pd(path, header_row_n=0)
        df = select_data(df) # select participant
        df = df.set_index('Anonymized ID')
        frame_df.append(df.iloc[:, :-1])
    print '{:55}:{:5}'.format(path.split('/')[-1], df.shape[0])

# filter by age
df_cog_measure = pd.concat(frame_df, axis=1).query('55 >= AGE >= 18')
print '='
print '{:55}:{:5}'.format('Number of included participants between age 18 - 55', df_cog_measure.shape[0])

# drop cases with more than 5 missings - listwise
null_cases_per_subj = np.sum(pd.isnull(df_cog_measure.iloc[:, 3:]).values, axis=1)
excludeIdx = np.where(null_cases_per_subj>5)
df_cog_measure = df_cog_measure.drop(df_cog_measure.index[excludeIdx])
print '='
print '{:55}:{:5}'.format('Number of participants selected', df_cog_measure.shape[0])


# next use the ID information to find the appropriate FC data
dict_data = {}
for cur in PATHS_FC:
    data_FC = np.load(cur)
    data_FC_include = []
    set_lab = cur.split('/')[-1].split('_')[3]
    for i, ID in enumerate(FC_subj):
        if ID in list(df_cog_measure.index):
            data_FC_include.append(data_FC[i, :])
    data_FC_include = np.array(data_FC_include)
    print '{:10}:{:5}'.format(set_lab, data_FC_include.shape[0])
    dict_data['FC_' + set_lab] = data_FC_include

# drop useless info
df_cog_measure = df_cog_measure.drop(['SubjectType', 'Visit'], axis=1)
print 'participants: ', df_cog_measure.shape[0]

'''
Preprocess data
'''

# impute outliers and missing values with mean, transform to z score
all_data = imputedata(df_cog_measure.values, 'mean') 

# save every numerical variables in z score aside from age
dict_data['Age']  = np.reshape(df_cog_measure.values[:, 0], newshape=(df_cog_measure.values[:, 0].shape[0], 1))

dict_data['MRIQ'] = all_data[:, 1:32]
dict_data['Motion_power']  = np.reshape(all_data[:, 32], newshape=(all_data[:, 32].shape[0], 1))
dict_data['Motion_Jenkinson']  = np.reshape(all_data[:, 33], newshape=(all_data[:, 33].shape[0], 1))
dict_data['CognitiveMeasures'] = all_data[:, 34:-1]
dict_data['CognitiveMeasures_labels'] = list(df_cog_measure.columns)[34:-1]
dict_data['Gender'] = np.reshape(df_cog_measure.values[:, -1], newshape=(df_cog_measure.values[:, -1].shape[0], 1))
dict_data['IDs'] = list(df_cog_measure.index)


# load labels and save
MRIQ_labels = []
with open('./references/8100_MRIQ_QuestionKeys.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        MRIQ_labels.append(row[-1])
dict_data['MRIQ_labels'] = MRIQ_labels

# FC data ROI labes
for p in PATHS_ROI:
    ROIlabs = list(np.load(p))
    dat_name = p.split('_')[3] + '_ROIs'
    dict_data[dat_name] = ROIlabs

# save all cognitive measures for later use. Need to be preprocessed
with open('./data/dict_SCCA_data_prepro_node-node.pkl', 'wb') as handle:
    pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
