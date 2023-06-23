#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:11:16 2023

@author: floriancurvaia
"""

import numpy as np

import time

import pandas as pd

import numba

import sys

start_time_0=time.time()

path_in='/data/homes/fcurvaia/features/'
path_out="/data/homes/fcurvaia/distances/"
im=str(sys.argv[1])
fn=path_in+im+".csv"



features=pd.read_csv(fn, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "structure", "pSmad2/3_2_Mean" ]) #




print("Read and load files --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()
#origin=np.load("/data/homes/fcurvaia/Spheres_fit/out_sphere_origin_"+im+".npy")
origin=np.load("/data/homes/fcurvaia/Spheres_fit/out_elli_origin_"+im+".npy")

features[["Centroid_x","Centroid_y", "Centroid_z"]]=features[["Centroid_x","Centroid_y", "Centroid_z"]]/[0.65,0.65,1]


feat_filt=features[(features.structure=='cells')]

del features

cell_pos=np.ascontiguousarray(np.array(feat_filt[[ "Centroid_z", "Centroid_y","Centroid_x"]]))

#inner_surface=np.load("/data/homes/fcurvaia/Spheres_fit/in_sphere_"+im+".npy")
inner_surface=np.load("/data/homes/fcurvaia/Spheres_fit/in_elli_"+im+".npy")

ed_id_in=np.array(np.where(inner_surface==True))


print("Prepare data --- %s seconds ---\n" % (time.time() - start_time_1))


# EUCLEDIAN DISTANCE
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

min_dists_in=np.empty((0,))
for i in range(0, cell_pos.shape[0],2000):
    if i+2000>cell_pos.shape[0]:
        j=cell_pos.shape[0]
    else:
        j=i+2000
    start_time_3=time.time()
    min_dists_in_i=np.sqrt(np.min(pz_dist(cell_pos[i:j], ed_id_in[0].astype("float64"), ed_id_in[1].astype("float64"), ed_id_in[2].astype("float64")), axis=1))
    min_dists_in=np.append(min_dists_in, min_dists_in_i)
    print("Get minimal inner distances --- %s seconds ---\n" % (time.time() - start_time_3))
del ed_id_in, inner_surface, min_dists_in_i
min_dists_in=np.stack(min_dists_in, axis=0)

#outer_surface=np.load("/data/homes/fcurvaia/Spheres_fit/out_sphere_"+im+".npy")
outer_surface=np.load("/data/homes/fcurvaia/Spheres_fit/out_elli_"+im+".npy")
ed_id_out=np.array(np.where(outer_surface==True))
min_dists_out=np.empty((0,))
for i in range(0, cell_pos.shape[0],2000):
    if i+2000>cell_pos.shape[0]:
        j=cell_pos.shape[0]
    else:
        j=i+2000
    start_time_3=time.time()
    min_dists_out_i=np.sqrt(np.min(pz_dist(cell_pos[i:j], ed_id_out[0].astype("float64"), ed_id_out[1].astype("float64"), ed_id_out[2].astype("float64")), axis=1))
    min_dists_out=np.append(min_dists_out, min_dists_out_i)
    print("Get minimal outer distances --- %s seconds ---\n" % (time.time() - start_time_3))
del ed_id_out, outer_surface, min_dists_out_i
min_dists_out=np.stack(min_dists_out, axis=0)

feat_filt["dist_in"]=min_dists_in
feat_filt["dist_out"]=min_dists_out
feat_filt["dist_out/in_out"]=min_dists_out/(min_dists_out+min_dists_in)

feat_filt.to_csv(path_out+im+"_w_dist_edge.csv", sep=",")

