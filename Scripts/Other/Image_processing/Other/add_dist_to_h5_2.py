#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:25:50 2023

@author: floriancurvaia
"""

import sys
import h5py
import numpy as np
import math
from abbott.h5_files import *
import time
import pandas as pd
import numba

#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B08_px+1076_py-0189.h5"

#h5_summary(fn)

start_time_0=time.time()
path_in_dist="/data/homes/fcurvaia/distances/"

#path_in='/data/active/sshami/20220716_experiment3_aligned/'
path_in="/data/active/fcurvaia/Segmented_files/"
im=str(sys.argv[1])
fn=path_in+im+".h5"

#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B08_px+1076_py-0189.h5"

@numba.njit('(float64[:,:,::1], uint16[:,:,::1], int64[::1], float64[::1])', parallel=True, fastmath=True)
def create_dist_array_2(d_array, c_array, label, distance):
    #lab_dist={A: B for A, B in zip(label, distance)}
    for j in numba.prange(len(label)):
        #idx=np.where(label==j)[0][0]
        lab=label[j]
        d=distance[j]
        for s1 in numba.prange(len(c_array)):
            for s2 in numba.prange(len(c_array[s1])):
                wh=np.where(c_array[s1, s2,:]==lab)
                for i in numba.prange(len(wh[0])):
                    d_array[s1, s2,:][wh[0][i]]+=d
    return d_array

with h5py.File(fn, "r+") as f:
    seg_cells=np.array(f["lbl_cells"])

dist_sph=np.zeros((seg_cells.shape))
feat_filt=pd.read_csv(path_in_dist+im+"_w_dist_sph_simp.csv", sep=",")
    
    
print("load files --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()


dist_sph=create_dist_array_2(dist_sph, seg_cells, feat_filt["Label"].to_numpy(), feat_filt["dist_out"].to_numpy() )
    
print("Iterate trhough cells --- %s seconds ---\n" % (time.time() - start_time_1))
    
start_time_2=time.time()
with h5py.File(fn, "r+") as f:
    h5_write_channel(f, dist_sph, "dist_sph_4")
    
print("Write new dataset --- %s seconds ---\n" % (time.time() - start_time_2))





