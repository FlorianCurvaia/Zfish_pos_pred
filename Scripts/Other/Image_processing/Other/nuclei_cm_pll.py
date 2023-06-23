#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 13:32:24 2023

@author: floriancurvaia
"""
import h5py

import numpy as np

from joblib import Parallel, delayed

#import time

from abbott.h5_files import *

#start_time_0=time.time()
#fn =h5py.File("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Fileserver/fcurvaia/results/20230124_experiment5_Shield_compressedB03_px-1662_py+1367.h5")

#seg_nuc=np.array(fn["lbl_nuc_raw3"])
#print("My way --- %s seconds ---\n" % (time.time() - start_time_0))

#start_time_1=time.time()

fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Fileserver/fcurvaia/results/20230124_experiment5_Shield_compressedB03_px-1662_py+1367.h5"

with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'nucleiRaw3'}):
        seg_nuc = to_numpy(dset)
        #print(dset)
#print("Shayan's way --- %s seconds ---\n" % (time.time() - start_time_1))


nuc_ids=range(1, np.max(seg_nuc)+1)


def get_cm(arr, nucleus):
    tu=np.unravel_index(np.where(arr.ravel()==nucleus),arr.shape)
    to=[]
    for i in range(3):
        to.append(np.sum(tu[i])/tu[i].shape[1])
    return to

results=Parallel(n_jobs=4)(delayed(get_cm)(seg_nuc, nuc) for nuc in nuc_ids)

if len(results[0])!=3:
    raise ValueError("There is not 3 coordinates in the results : "+ str(results[0]))
    
with open("Centers_of_mass_nuclei.tsv", "w") as out:
    header="Nucleus"+"\t"+"x"+"\t"+"y"+"\t"+"z"+"\n"
    out.write(header)
    for i, c_m in zip(nuc_ids, results):
        to_write=str(i)+"\t"+str(c_m[1])+"\t"+str(c_m[2])+"\t"+str(c_m[0])+"\n"
        out.write(to_write)
    embryo_cm=get_cm(np.where(seg_nuc>=1, 1, 0), 1)
    to_write=str(0)+"\t"+str(embryo_cm[1])+"\t"+str(embryo_cm[2])+"\t"+str(embryo_cm[0])+"\n"
    out.write(to_write)

#start_time_2=time.time()
#get_cm(seg_nuc, 1)
#print("new func --- %s seconds ---\n" % (time.time() - start_time_2))