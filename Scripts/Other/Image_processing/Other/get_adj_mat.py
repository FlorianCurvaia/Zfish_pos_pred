#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:37:40 2023

@author: floriancurvaia
"""

import itk
import numpy as np
import sys
import h5py
from abbott.neighborhood_matrix_parallel import weighted_anisotropic_touch_matrix #, neighborhood_matrix
from abbott import itk_image
from abbott.conversions import *
import time
#from neigh_mat import adjacency_matrix




start_time_0=time.time() 
path_in='/data/active/sshami/20220716_experiment3_aligned/'
path_out="/data/homes/fcurvaia/Spheres_fit/"
im=str(sys.argv[1]) #"B08_px+1076_py-0189"#str(sys.argv[1])
fn=path_in+im+".h5"

with h5py.File(fn) as f:
    seg_cells=np.array(f["lbl_cells"])
    
print("load file --- %s seconds ---\n" % (time.time() - start_time_0))
    
start_time_1=time.time() 
seg_cells=to_itk(seg_cells)
print("to_itk --- %s seconds ---\n" % (time.time() - start_time_1))

start_time_2=time.time() 
seg_cells=to_labelmap(seg_cells)
print("to_labelmap --- %s seconds ---\n" % (time.time() - start_time_2))


start_time_3=time.time() 
def get_adjacency_matrix(
lbl_map: itk.LabelMap
) -> np.array:
    nm = weighted_anisotropic_touch_matrix(
        itk.GetArrayFromImage(itk_image.to_labelimage(lbl_map)).astype(np.int32)
    )
    return(nm)
adj_mat=get_adjacency_matrix(seg_cells)
print("get adjacency matrix --- %s seconds ---\n" % (time.time() - start_time_3))
adj_mat=np.delete(adj_mat, 0, 0)
adj_mat=np.delete(adj_mat, 0, 1)
np.fill_diagonal(adj_mat, 0)

np.save(path_out+im+"_adj_mat", adj_mat.astype("bool"))

"""
start_time_4=time.time() 
def get_adjacency_matrix_2(
lbl_map: itk.LabelMap
) -> np.array:
    nm = neighborhood_matrix(
        itk.GetArrayFromImage(itk_image.to_labelimage(lbl_map)).astype(np.int32)
    )
    return(nm)
adj_mat_2=get_adjacency_matrix(seg_cells)
print("get adjacency matrix 2 --- %s seconds ---\n" % (time.time() - start_time_4))

np.save(path_out+im+"_adj_mat_2", adj_mat_2)

start_time_4=time.time() 
adj_mat_2=adjacency_matrix(itk.GetArrayFromImage(itk_image.to_labelimage(seg_cells)).astype(np.int32))
print("get adjacency matrix 2 --- %s seconds ---\n" % (time.time() - start_time_4))

np.save(path_out+im+"_adj_mat_3", adj_mat_3)
"""