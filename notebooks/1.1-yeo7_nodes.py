from nilearn.regions import connected_label_regions
import numpy as np
import nibabel as nib

def hemisphere_split(atlas, names):

    labels = np.unique(atlas)
    # split to left and right first
    # find the middle index and create hemisphere masks
    middle_ind = (atlas.shape[0] - 1) // 2
    left_atlas = atlas.copy()
    left_atlas[middle_ind:, ...] = 0
    right_atlas = atlas.copy()
    right_atlas[middle_ind:, ...] = 0
    # split those and relabel
    new_label = 0
    new_atlas = atlas.copy()
    new_names = [names[0]]
    for label, name in zip(labels[1:], names[1:]):
        new_label += 1
        left_elements = (left_atlas == label).sum()
        right_elements = (right_atlas == label).sum()
        n_elements = float(left_elements + right_elements)
        if (left_elements / n_elements < 0.05 or
                right_elements / n_elements < 0.05):
            new_atlas[atlas == label] = new_label
            new_names.append(name)
            continue
        new_atlas[right_atlas == label] = new_label
        new_names.append('Left-' + name)
        new_label += 1
        new_atlas[left_atlas == label] = new_label
        new_names.append('Right-' + name)
    return new_atlas, new_names

yeo7 = '~/Patterns-of-Thought/references/Yeo7LiberalMask_lables.nii.gz'

yeo7_nii = nib.load(yeo7)
yeo7_names = ['Background', 'Visual', 'Somatomotor', 'DorsalAttention',
                'VentralAttention', 'Limbic', 'Frontoparietal', 'Default']
yeo7_atlas = yeo7_nii.get_data()

new_atlas, new_names = hemisphere_split(yeo7_atlas, yeo7_names)
yeo7_LR_nii = nib.Nifti1Image(new_atlas, affine=yeo7_nii.affine, header=yeo7_nii.header)

# split to smaller regions
region_labels = connected_label_regions(yeo7_LR_nii, connect_diag=False, min_size=100)
region_labels.to_filename('~/Patterns-of-Thought/references/yeo7_LR_57clusters.nii.gz')
