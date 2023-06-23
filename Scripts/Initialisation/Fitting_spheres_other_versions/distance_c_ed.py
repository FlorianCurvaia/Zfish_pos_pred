#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:38:56 2023

@author: floriancurvaia
"""
import numpy as np

#from math import sqrt, acos

import time

import h5py

from joblib import Parallel, delayed

from abbott.h5_files import *

from scipy.spatial.distance import cdist

import pandas as pd

from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from math import pi

import numba

start_time_0=time.time()
fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/B08_px+1076_py-0189.csv"

features=pd.read_csv(fn, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "structure", "pSmad2/3_2_Mean" ]) #

#seg_cell=np.load("seg_cell.npy")

edges_2=np.load("Arrays/edges_2.npy")

print("Read and load files --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()
origin=[517.6166872756335806*0.65, 534.9993432631106316*0.65, 513.8579595065110652]

#features[["Centroid_x","Centroid_y", "Centroid_z"]]=features[["Centroid_x","Centroid_y", "Centroid_z"]]-[origin[0],origin[1],origin[2]]
#features[["Centroid_x","Centroid_y"]]=features[["Centroid_x","Centroid_y"]]*0.65
features[["Centroid_x","Centroid_y", "Centroid_z"]]=features[["Centroid_x","Centroid_y", "Centroid_z"]]/[0.65,0.65,1]


feat_filt=features[(features.structure=='cells')]

cell_pos=np.array(feat_filt[[ "Centroid_z", "Centroid_y","Centroid_x"]])


ed_id=np.array(np.where(edges_2==True)).T

ed_id_test=ed_id[0:2000,:]

cell_pos_test=cell_pos[0:1500]

print("Prepare data --- %s seconds ---\n" % (time.time() - start_time_1))

start_time_2=time.time()
min_dists_test=np.sqrt(np.min(cdist(cell_pos, ed_id_test.astype("float64"), "sqeuclidean"), axis=1))
print("Get minimal distances --- %s seconds ---\n" % (time.time() - start_time_2))

print(min_dists_test)

# EUCLEDIAN DISTANCE Source: https://stackoverflow.com/questions/49664523/how-can-i-speed-up-closest-point-comparison-using-cdist-or-tensorflow
@numba.njit('(float64[:,::1], float64[::1], float64[::1], float64[::1])', parallel=True, fastmath=True)
def pz_dist(p_array, x_flat, y_flat, z_flat):
    m = p_array.shape[0]
    n = x_flat.shape[0]
    d = np.empty(shape=(m, n), dtype=np.float64)
    for i in numba.prange(m):
        p1 = p_array[i, :]
        for j in range(n):
            _x = x_flat[j] - p1[0]
            _y = y_flat[j] - p1[1]
            _z = z_flat[j] - p1[2]
            _d = _x**2 + _y**2 + _z**2
            d[i, j] = _d
    return d

start_time_3=time.time()
min_dists_test_2=np.sqrt(np.min(pz_dist(np.ascontiguousarray(cell_pos), ed_id_test.T[0].astype("float64"), ed_id_test.T[1].astype("float64"), ed_id_test.T[2].astype("float64")), axis=1))
print("Get minimal distances 2 --- %s seconds ---\n" % (time.time() - start_time_3))
feat_filt["dist_to_edge"]=min_dists_test_2

feat_filt.to_csv("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/B08_px+1076_py-0189_w_dist_edge.csv", sep=";")




