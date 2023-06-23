#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:05:28 2023

@author: floriancurvaia
"""



import h5py

from abbott.h5_files import *

import numpy as np

from pathlib import Path

import math

import random

import time

import argparse

start_time_0=time.time()

def sphereFit(coords):
    #   Assemble the A matrix
    spX = np.squeeze(coords[2])#*0.65
    spY = np.squeeze(coords[1])#*0.65
    spZ = np.squeeze(coords[0])
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f, rcond=None)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2] 

def get_middle_sphere(edges, n_points=5000):
    random.seed(42)
    rand_id_z, rand_id_x, rand_id_y=np.unravel_index(
        random.sample(
            np.where(edges.ravel()==True)[0].tolist(),
            k=n_points), 
        edges.shape)
    indices=(rand_id_z, rand_id_y, rand_id_x)
    r, x0, y0, z0 = sphereFit(indices)
    return (r, x0, y0, z0)

CLI=argparse.ArgumentParser()
CLI.add_argument('idx', type=int)
CLI.add_argument(
  "--flds",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)

args = CLI.parse_args()


ref_cycle=1
n_cycles=6
cycles_to_pre_align=list(set(range(n_cycles))-set([ref_cycle]))
#D05 weird
#im="B07_px+1257_py-0474" 
#B07_px+1257_py-0474 B08_px+1076_py-0189 C04_px-0816_py-1668 B03_px-0545_py-1946 B06_px-0493_py+0146 C05_px+0198_py+1683
#D06_px-1055_py-0118 D05_px-0738_py-1248 B05_px+1522_py-1087

#fn_in='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/'+im+".h5"

#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/tmp_fcurvaia/"+im+".h5"

path_in_emb='/data/active/fcurvaia/Segmented_files/'

#fn_out='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Compressed_files/'+im+"_shifted.h5"


path_in=Path(args.flds[0]) 
image=list(path_in.glob("*.h5*"))[args.idx].name
#path_out=path_in.parent / (path_in.name+"_aligned")
path_out=Path("/data/active/fcurvaia/Compressed_files_aligned")
if not path_out.exists():
    path_out.mkdir()

fn=path_in / image
fn_out=path_out / image
path_in_emb=Path(path_in_emb)
fn_emb=path_in_emb / image

with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain':'DAPI', 'level': 1, 'cycle': ref_cycle}):
        cycle = dset.attrs['cycle']
        scale = dset.attrs['element_size_um']
        DAPI_ref=to_numpy(dset)
        
   
with h5py.File(fn_emb) as f:
    for dset in h5_select(f, {'stain':'embryo','level': 1}):
        emb_seg=to_numpy(dset)


#DAPI_ref=all_datasets_ref["DAPI"]   
DAPI_ref_bool=DAPI_ref>np.quantile(DAPI_ref, 0.99)

r_ref, x_ref, y_ref, z_ref = get_middle_sphere(DAPI_ref_bool, 25000)
c_ref=np.array([x_ref, y_ref, z_ref])


for cycle in cycles_to_pre_align:
    with h5py.File(fn) as f:
        for dset in h5_select(f, {'stain':'DAPI','level': 1, 'cycle': cycle}):
            DAPI_c=to_numpy(dset)

    
    DAPI_c_bool=DAPI_c>np.quantile(DAPI_c, 0.99)
    
    r_c, x_c, y_c, z_c = get_middle_sphere(DAPI_c_bool, 25000)
    
    
    x_shift= int(math.floor(x_ref[0]-x_c[0]))
    y_shift= int(math.floor(y_ref[0]-y_c[0]))
    
    
    DAPI_ref[:, :, -abs(y_shift):]=0
    
    DAPI_ref[:, -abs(x_shift):, :]=0
    
    emb_seg[:, :, -abs(y_shift):]=0
    
    emb_seg[:, -abs(x_shift):, :]=0
    


    
                
with h5py.File(fn_out, "a") as f_out:
    with h5py.File(fn_emb) as f_emb:
        for dset in h5_select(f_emb, {'stain': "embryo", 'level': 1}):
            h5_write_channel(f_out, emb_seg, dset.name, copy_from=dset)
        

print("Run Time --- %s seconds ---\n" % (time.time() - start_time_0))



