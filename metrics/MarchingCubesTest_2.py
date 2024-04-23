# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 23:53:28 2024

@author: josef
"""

import nibabel as nib
import numpy as np
import skimage

#%% Read a volumen
nii_img = nib.load('image.nii.gz').get_fdata()
nii_msk = nib.load('mask.nii.gz').get_fdata()

print(nii_msk.shape, nii_img.shape)
print(np.max(nii_msk), np.max(nii_img))

#%% 
verts, faces, normals, values = skimage.measure.marching_cubes(nii_img, method='lorensen')
print(verts.shape, faces.shape, normals.shape, values.shape)

#%%