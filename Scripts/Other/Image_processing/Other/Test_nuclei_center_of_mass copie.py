#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:49:33 2023

@author: floriancurvaia
"""
import h5py

import numpy as np

from scipy.ndimage import center_of_mass

from joblib import Parallel, delayed


f =h5py.File("/Volumes/Elements/Test_images/20230124_experiment5_Shield_compressedB03_px-1662_py+1367.h5")

seg_nuc=np.array(f["lbl_nuc_raw4"])

centers_of_mass=dict()
nuc_ids=list(range(np.min(seg_nuc), np.max(seg_nuc)))

#for nuc in nuc_ids:
#    masked = np.where(seg_nuc==nuc, 1, 0)
#    centers_of_mass[nuc]=center_of_mass(masked)

def get_cm(arr, nucleus):
    masked = np.where(arr==nucleus, 1, 0)
    return(center_of_mass(masked))

results=Parallel(n_jobs=10)(delayed(get_cm)(seg_nuc, nuc) for nuc in nuc_ids)
if len(results[0])!=3:
    raise ValueError("There is not 3 coordinates in the results : "+ str(results[0]))
    
with open("Centers_of_mass_nuclei.csv", "w") as out:
    header="Nucleus"+"\t"+"x"+"\t"+"y"+"\t"+"z"+"\n"
    for i, c_m in zip(nuc_ids, results):
        to_write=str(i)+"\t"+str(c_m[1])+"\t"+str(c_m[2])+"\t"+str(c_m[0])+"\n"
        out.write(to_write)
        