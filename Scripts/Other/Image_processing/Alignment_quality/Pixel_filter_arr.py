#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:11:21 2023

@author: floriancurvaia
"""
import h5py
import numpy as np
import pandas as pd
from abbott.h5_files import *
import time
#from functools import reduce
#import itertools
import numba
import argparse
#from pathlib import Path
#import napari
#from skimage.measure import regionprops_table
from tqdm import tqdm
#import matplotlib.pyplot as plt
#import seaborn as sns
#from scipy.stats import pearsonr
import nest_asyncio
nest_asyncio.apply()
#from matplotlib.widgets import Slider

start_time_0=time.time()
CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--image",  # name on the CLI - drop the `--` for positional/required parameters
  nargs=1,  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)
CLI.add_argument(
  "--cycles",  # name on the CLI - drop the `--` for positional/required parameters
  nargs=1,  # 0 or more values expected => creates a list
  type=int,
  default=[],  # default if nothing is provided
)

args = CLI.parse_args()


im=args.image[0]
cycle=args.cycles[0]

path_in='/data/homes/fcurvaia/distances/'

fn_csv=path_in+im+"_corr_dapi.csv"

corr_nuc=pd.read_csv(fn_csv, sep=",", index_col=False)
corr_nuc.fillna(0, inplace=True)

fn_im="/data/active/sshami/20220716_experiment3_aligned/"+im+".h5"


with h5py.File(fn_im) as f:
    nucs=h5_select(f, {'stain': "nucleiRaw"})[0]
    nucs=to_numpy(nucs)

np.save(path_in+im+"_nuc_seg.npy", nucs)
print("load files and argument --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()



@numba.njit(parallel=True)
def isin(a, b):
    out=np.empty(a.shape, dtype=numba.boolean)
    b = set(b)
    for i in numba.prange(a.shape[0]):
        for j in numba.prange(a.shape[1]):
            out[i, j]=a[i, j] in b
            """
            if a[i] in b:
                out[i]=True
            else:
                out[i]=False
            """
    return out


@numba.jit(nopython=True, parallel=True, fastmath=True)
def where_nuc_bad(nucs, nucs_to_rm):
    out=np.empty(nucs.shape, dtype=numba.boolean)
    for i in numba.prange(nucs.shape[0]):
        out[i]=isin(nucs[i], nucs_to_rm)
    return(out)
        

for i in range(cycle):
    to_rm=corr_nuc["Label"].loc[corr_nuc[str(i)]<0.9].to_numpy()
    where_to_rm=where_nuc_bad(nucs, to_rm)
    np.save(path_in+im+"_nuc_bad_"+str(i)+".npy", where_to_rm)
    
print("find to remove --- %s seconds ---\n" % (time.time() - start_time_1))












