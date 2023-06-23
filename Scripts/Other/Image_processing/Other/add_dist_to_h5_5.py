#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:25:50 2023

@author: floriancurvaia
"""

#import sys
import h5py
import numpy as np
#import math
from abbott.h5_files import *
import time
import pandas as pd
import numba
import argparse
from multiprocessing import Pool

#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B08_px+1076_py-0189.h5"

#h5_summary(fn)

start_time_0=time.time()
CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--image",  # name on the CLI - drop the `--` for positional/required parameters
  nargs=1,  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)
CLI.add_argument(
  "--colnames",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)

args = CLI.parse_args()
path_in_dist="/data/homes/fcurvaia/distances/"

path_in='/data/active/sshami/20220716_experiment3_aligned/'
#path_in="/data/active/fcurvaia/Segmented_files/"
im=args.image[0]
fn=path_in+im+".h5"

#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B08_px+1076_py-0189.h5"

@numba.njit('(float64[:,:,::1], uint16[:,:,::1], int64[::1], float64[::1])', parallel=True, fastmath=True)
def create_dist_array_2(d_array, c_array, label, distance):
    #lab_dist={A: B for A, B in zip(label, distance)}
    for j in numba.prange(len(label)):
        #idx=np.where(label==j)[0][0]
        lab=label[j]
        d=distance[j]
        for s in numba.prange(len(c_array)):
            wh=np.where(c_array[s]==lab)
            for i in numba.prange(len(wh[0])):
                d_array[s][wh[0][i], wh[1][i]]=d
    return d_array

with h5py.File(fn, "r+") as f:
    seg_cells=np.array(f["lbl_cells"])

#fn_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/B08_px+1076_py-0189_w_dist_sph_simp.csv"
#feat_filt=pd.read_csv(fn_dist, sep=",")

feat_filt=pd.read_csv(path_in_dist+im+"_w_dist_sph_simp.csv", sep=",")
    
    
#print("load files --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()

#arrays_to_write=[]
#data_to_write=[]
pool_arguments=[]
for col in args.colnames:
    arr_to_write=np.zeros((seg_cells.shape))
    #arr_to_write=np.zeros((3,10,10))
    #arrays_to_write.append(arr_to_write)
    #data_to_write.append(feat_filt[col].to_numpy())
    pool_arguments.append((arr_to_write, seg_cells, feat_filt["Label"].to_numpy(), feat_filt[col].to_numpy()))


pool=Pool()
#arrays_to_write=create_dist_array_2(numba.typed.List(arrays_to_write), seg_cells, feat_filt["Label"].to_numpy(), numba.typed.List(data_to_write) )
arrays_to_write=list(pool.starmap(create_dist_array_2, pool_arguments))
"""
for col in args.colnames:
    
    start_time_1_1=time.time()
    
    
    arr_to_write=create_dist_array_2(arr_to_write, seg_cells, feat_filt["Label"].to_numpy(), feat_filt[col].to_numpy() )
    np.save(path_in_dist+im+"_"+col, arr_to_write)
    arrays_to_write.append(arr_to_write)
    print("Iterate trhough cells 1 col --- %s seconds ---\n" % (time.time() - start_time_1_1))
"""
print("Iterate trhough cells all cols --- %s seconds ---\n" % (time.time() - start_time_1))    
start_time_2=time.time()
with h5py.File(fn, "r+") as f:
    for arr, col in zip(arrays_to_write, args.colnames):
        h5_write_channel(f, arr, col)
        
print("Write new dataset --- %s seconds ---\n" % (time.time() - start_time_2))





