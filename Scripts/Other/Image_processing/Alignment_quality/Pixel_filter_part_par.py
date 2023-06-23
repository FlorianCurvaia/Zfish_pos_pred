#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:38:58 2023

@author: floriancurvaia
"""


import h5py
import numpy as np

from abbott.h5_files import *
import time

#import plotly.graph_objects as go

import numba
import argparse


start_time_0=time.time()
CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--image",  # name on the CLI - drop the `--` for positional/required parameters
  nargs=1,  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)


args = CLI.parse_args()
path_in_dist='/data/homes/fcurvaia/features/' #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/"#'/data/homes/fcurvaia/features/'
path_out_dist="/data/homes/fcurvaia/distances/" #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/"#/data/homes/fcurvaia/distances/"
#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/"#"/data/active/fcurvaia/Segmented_files/"#'/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/'#B08_px+1076_py-0189.h5'
#fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B07_px+1257_py-0474.h5'
path_in='/data/active/sshami/20220716_experiment3_aligned/'
path_out="/data/homes/fcurvaia/Spheres_fit/"
im=args.image[0] #"B08_px+1076_py-0189"#str(sys.argv[1])
fn=path_in+im+".h5"
#fn_dist=path_in_dist+im+".csv"
n_cycles=6
dapi_cycles=[]
with h5py.File(fn) as f:
    for c in range(n_cycles):
        for dset in h5_select(f, {'stain': 'DAPI', "cycle":c, "level":1}):
            Nucleiseg = to_numpy(dset)
            dapi_cycles.append(Nucleiseg)
            


with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'cells'}):
        seg_cells = to_numpy(dset)

del dset, c, Nucleiseg

labels = np.unique(seg_cells)[1:]
#print(labels)

print("load files and argument --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()

dapi_cycles_not_1=dapi_cycles[:1] + dapi_cycles[1+1:]
dapi_cycle_1=dapi_cycles[1]
del dapi_cycles
#@numba.njit('(uint16[:,:,::1], types.List(float64[:,:,::1], reflected=True), uint16[::1])', parallel=True, fastmath=True)
@numba.jit(nopython=True, parallel=True, fastmath=True)
def clean_pixels(cells, ref_cycle, cycle, labels):
    for label in labels:
        pixs_cell=np.where(cells==label)
        pixels1 = ref_cycle[pixs_cell[0],:,:]
        pixels2 = cycle[pixs_cell[0],:,:]
        pixels1 = pixels1[:, pixs_cell[1], :]
        pixels2 = pixels2[:, pixs_cell[1], :]
        pixels1 = pixels1[:, :, pixs_cell[2]]
        pixels2 = pixels2[:, :, pixs_cell[2]]
        #data=np.array([pixels1.flatten(), pixels2.flatten()])
        data=np.stack((pixels1.flatten(), pixels2.flatten()))
        cov=np.cov(data)
        corr=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
        pixs_cell, data, pixels1, pixels2, cov=0,0,0,0,0
        if corr <0.9:
            for i in numba.prange(cells.shape[0]):
                for j in numba.prange(cells.shape[1]):
                    cells[i][ j, :][np.where(cells[i][ j, :]==label)]=0
    return(cells)

for c in dapi_cycles_not_1:
    seg_cells=clean_pixels(seg_cells,dapi_cycle_1, c, labels)
    
print("clean pixels --- %s seconds ---\n" % (time.time() - start_time_1))

start_time_2=time.time()
with h5py.File(fn, "r+") as f:
    dset = h5_select(f, {'stain': 'cells'})
    h5_write_channel(f, seg_cells, "cells_clean", copy_from=dset[0])
        
    
print("clean pixels --- %s seconds ---\n" % (time.time() - start_time_2))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    