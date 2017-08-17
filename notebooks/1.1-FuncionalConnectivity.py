RS_NII_PATH = '/groups/labs/semwandering/CPAC/NKI/results/pipeline_NKI/*/functional_mni_other_resolutions_smooth/_scan_rest_1400_rest_1400/_compcor_ncomponents_5_selector_pc10.linear0.wm0.global0.motion1.quadratic0.gm0.compcor1.csf0/_bandpass_freqs_0.009.0.1/_fwhm_6/bandpassed_demeaned_filtered_warp_maths.nii.gz'

ID_loc = 8

ATLAS_NII_PATH = '/home/h/hw1012/Masks/Yeo7LiberalMask/*.nii.gz'

out_filename_root = '/home/h/hw1012/Project_NKI/CCA/Data/data_RSFC_Yeo7'

out_atlas_lable = '/home/h/hw1012/Project_NKI/CCA/Data/Yeo7LiberalMask_lables.nii.gz'

##################################################################

import glob
import os
import sys

import numpy as np

import nibabel as nib
from nilearn.image import index_img, resample_img
from nilearn.input_data import NiftiLabelsMasker

# get all resting state files
rs_niis = sorted(glob.glob(RS_NII_PATH))

# generate labels
atlas_nii = sorted(glob.glob(ATLAS_NII_PATH))

#ROI labels
atlas_names = [a.split('.nii.gz')[0].split('/')[-1] for a in atlas_nii]
atlas_names = np.array(atlas_names)

print 'ROI text labels in "Data" folder'

print 'Resample ROI masks to match the data'

tmp_nii_path = rs_niis[0]
tmp_nii = nib.load(tmp_nii_path)

label_atlas = np.zeros(tmp_nii.shape[:3], dtype=np.int)
for i_roi, cur_roi in enumerate(atlas_nii):
    # reshape the data (put the mask on the particiapnt's data, matching the coordinates and shapes)
    re_cur_roi = resample_img(
        img=cur_roi,
        target_affine=tmp_nii.get_affine(),
        target_shape=tmp_nii.shape[:3],
        interpolation='nearest'
    )

    # binarize the data
    cur_data = re_cur_roi.get_data()
    if cur_data.ndim > 3: cur_data = cur_data[...,0] # debug
    cur_data_bin = np.array(cur_data > 0, dtype=np.int)
    label_atlas[cur_data_bin > 0] = i_roi + 1

label_atlas_nii = nib.Nifti1Image(
    label_atlas,
    affine=tmp_nii.get_affine(),
    header=tmp_nii.get_header()
)

label_atlas_nii.to_filename(out_atlas_lable)
print 'label masker generated. fitting the data.'

masker = NiftiLabelsMasker(labels_img=label_atlas_nii, standardize=True,
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
np.save(out_filename_root+'_ROIs', atlas_names)
np.save(out_filename_root, corr_mat_vect_array)
np.save(out_filename_root + '_pptID', np.array(ppt_id))

# FC labels
reg_reg_names = [atlas_names[a] + ' vs ' + atlas_names[b] for (a,b) in zip(triu_inds[0], triu_inds[1])]
np.save(out_filename_root+'_keys', np.array(reg_reg_names))

print 'Job done!'
