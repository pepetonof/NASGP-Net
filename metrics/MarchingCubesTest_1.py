# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:44:17 2024

@author: josef
"""

import nibabel as nib
import numpy as np

#%%Read a volumen
nii_img = nib.load('image.nii.gz').get_fdata()
nii_msk = nib.load('mask.nii.gz').get_fdata()

# print(nii_msk.shape, nii_img.shape)
# print(np.max(nii_msk), np.max(nii_img))

#%%Surface Lookup Table for Marching Squares

def SampleGrid(G:np.ndarray, res_x:np.int32, res_y:np.int32, iso:int):
    m, n = G.shape
    p = int(m/res_x)
    q = int(n/res_y)
    
    print(m,n,p,q)
    F = np.zeros((p,q), dtype=np.bool)
    coord = np.zeros((p,q), dtype=object)
    for i in range(p):
        for j in range(q):
            # print(i*res_x, j*res_y)
            if G[i*res_x, j*res_y]>iso:
                F[i,j] = 0
                # print('Zero\n')
            else:
                F[i,j] = 1
                # print('One\n')
            coord[i,j]=(i*res_x, j*res_y)
                
    return F, coord, p, q

def March(F, coord, p, q):
    gamma = []
    for i in range(p-1):
        for j in range(p-1):
            a = coord[i,j]
            b = coord[i, j+1]
            c = coord[i+1, j+1]
            d = coord[i+1,j]
            k = 8*F[i,j]+4*F[i,j+1]
    

#%%
G = nii_img[:,:,70]
F, coord, p, q = SampleGrid(G, 32, 32, 400)





# def MarchingSquares(grid:np.ndarray, res:tuple, iso:int):
    