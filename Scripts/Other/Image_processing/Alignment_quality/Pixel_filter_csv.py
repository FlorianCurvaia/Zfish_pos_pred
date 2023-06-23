#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:47:56 2023

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
#import argparse
#from pathlib import Path
#import napari
#from skimage.measure import regionprops_table
from tqdm import tqdm
import matplotlib.pyplot as plt
#import seaborn as sns
#from scipy.stats import pearsonr
import nest_asyncio
nest_asyncio.apply()
from matplotlib.widgets import Slider

start_time_0=time.time()
im="B07_px+1257_py-0474" #C04_px-0816_py-1668 B07_px+1257_py-0474
cycles=6

path_in='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/'

fn_csv=path_in+im+"_corr_dapi.csv"

corr_nuc=pd.read_csv(fn_csv, sep=",", index_col=False)
corr_nuc.fillna(0, inplace=True)

"""
fn_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/"+im+".h5"


with h5py.File(fn_im) as f:
    nucs=h5_select(f, {'stain': "nucleiRaw"})[0]
    nucs=to_numpy(nucs)
"""
nucs=np.load(path_in+im+"_nuc_seg.npy")
print("load files and argument --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()

cycles_bad=[]
for i in range(cycles):
    cycles_bad.append(np.load(path_in+im+"_nuc_bad_"+str(i)+".npy"))
        
        
where_to_rm=cycles_bad[4]
print("find to remove --- %s seconds ---\n" % (time.time() - start_time_1))

fig1=plt.figure(1)
plt.clf()
idx0 = 195
m = plt.imshow(nucs[idx0], vmin=0, vmax=1)
n = plt.imshow(where_to_rm[idx0], vmin=0, vmax=1, alpha=0.75, cmap="magma")


axidx = plt.axes([0.25, 0, 0.65, 0.03])
slidx = Slider(axidx, 'z-stack', 0, nucs.shape[0]-1, valinit=idx0, valfmt='%d')

def update(val):
    idx = slidx.val
    m.set_data(nucs[int(idx)])
    n.set_data(where_to_rm[int(idx)])
    fig1.canvas.draw_idle()
slidx.on_changed(update)

plt.show()