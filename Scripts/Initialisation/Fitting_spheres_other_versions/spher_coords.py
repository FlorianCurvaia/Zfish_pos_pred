#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:03:54 2023

@author: floriancurvaia
"""

import numpy as np

#from math import sqrt, acos

#import time

import h5py

from joblib import Parallel, delayed

from abbott.h5_files import *


fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/results/20230124_experiment5_Shield_compressed/B03_px-1662_py+1367.h5"



with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'nucleiRaw3'}):
        seg_nuc = to_numpy(dset)


nuc_ids=range(1, np.max(seg_nuc)+1)


def get_cm(arr, nucleus):
    tu=np.unravel_index(np.where(arr.ravel()==nucleus),arr.shape)
    to=[]
    for i in range(3):
        to.append(np.sum(tu[i])/tu[i].shape[1])
    return to



results=Parallel(n_jobs=5)(delayed(get_cm)(seg_nuc, nuc) for nuc in nuc_ids)

results=np.array(results)

results[:,[0,1,2]]=results[:,[1,2,0]]

origin=get_cm(np.where(seg_nuc>=1, 1, 0), 1)

results=results-[origin[1],origin[2],origin[0]]


def appendSpherical_np(xyz): #from https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    ptsnew = np.hstack((xyz, np.empty(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2])
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

results=appendSpherical_np(results)



if results.shape[1]!=6:
    raise ValueError("There is not 6 coordinates in the results : "+ str(results[0]))
    
with open("Centers_of_mass_nuclei_spc.tsv", "w") as out:
    header="Nucleus"+"\t"+"x"+"\t"+"y"+"\t"+"z"+"\t"+"r"+"\t"+"theta"+"\t"+"phi"+"\n"
    out.write(header)
    for i, c_m in zip(nuc_ids, results):
        to_write=str(i)+"\t"+"\t".join(map(str, c_m))+"\n"
        out.write(to_write)



"""
start_time_0=time.time()
X=[]
Y=[]
Z=[]

with open("Centers_of_mass_nuclei.tsv", "r") as f:
    for row in f:
        line=row.strip().split("\t")
        if line[0]=="Nucleus":
            pass
        elif line[0]=="0":
            Ox,Oy,Oz=float(line[1]), float(line[2]), float(line[3])
        else:
            X.append(float(line[1])-Ox)
            Y.append(float(line[2])-Oy)
            Z.append(float(line[3])-Oz)

print("Read file --- %s seconds ---\n" % (time.time() - start_time_0))

r = lambda x, y, z: sqrt(x**2+y**2+z**2)

theta = lambda r, z: acos(z/r)

phi = lambda x, y : np.sign(y)*acos(x/sqrt(x**2+y**2))


tog=np.stack((X, Y, Z), axis=-1)


tog_w_sph=appendSpherical_np(tog)


"""