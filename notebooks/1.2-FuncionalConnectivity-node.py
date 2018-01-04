#!/usr/bin/python
# -*- coding: utf-8 -*-
#$ -o /scratch/$HOME/logs
#$ -e /scratch/$HOME/logs
#$ -N maskTS

# your nilearn path
NILEARN_PATH = '~/nilearn'

RS_NII_PATH = '/groups/labs/semwandering/CPAC/NKI/results/pipeline_NKI/*/functional_mni_other_resolutions_smooth/_scan_rest_1400_rest_1400/_compcor_ncomponents_5_selector_pc10.linear0.wm0.global0.motion1.quadratic0.gm0.compcor1.csf0/_bandpass_freqs_0.009.0.1/_fwhm_6/bandpassed_demeaned_filtered_warp_maths.nii.gz'

ID_loc = 8

out_filename_root = '~Patterns-of-Thought/data/interim/data_RSFC_Yeo7_nodes'

atlas_lable = '~/Patterns-of-Thought/references/yeo7_LR_57clusters.nii.gz'

##################################################################

import glob
import os
import sys

import numpy as np

# add nilearn to the cluster system directories so we can use nilearn functions
sys.path.append(NILEARN_PATH)
import nibabel as nib
from nilearn.image import index_img, resample_img
from nilearn.input_data import NiftiLabelsMasker

# get all resting state files
rs_niis = sorted(glob.glob(RS_NII_PATH))

masker = NiftiLabelsMasker(labels_img=atlas_lable, standardize=True,
                           memory='nilearn_cache', verbose=0)
masker.fit()

# caculate functional connectivity
corr_mat_vect_list = []
ind_list = []
ppt_id = []
for i_rs_img, rs_img in enumerate(rs_niis):

    rs_reg_ts = masker.transform(rs_img)
    corr_mat = np.corrcoef(rs_reg_ts.T)
    triu_inds = np.triu_indices(corr_mat.shape[0], 1)
    corr_mat_vect = corr_mat[triu_inds]
    # unchaged timeseries's connectivity will be one
    corr_mat_vect[np.isnan(corr_mat_vect)] = 1
    # save for later
    print('%i/%i: %s' % (i_rs_img + 1, len(rs_niis), rs_img))
    corr_mat_vect_list.append(corr_mat_vect)
    cur_subj = rs_img.split('/')[ID_loc].split('_')[0] # this bit can be better
    ppt_id.append(cur_subj)

corr_mat_vect_array = np.array(corr_mat_vect_list)

print(corr_mat_vect_array.shape)

np.save(out_filename_root, corr_mat_vect_array)
np.save(out_filename_root + '_pptID', np.array(ppt_id))
#np.save(out_filename_root+'_ROIs', atlas_names)
print 'Job done!'
