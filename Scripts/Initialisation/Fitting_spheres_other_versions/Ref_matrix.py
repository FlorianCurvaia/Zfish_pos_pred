#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:23:45 2023

@author: floriancurvaia
"""

import numpy as np

import time

import pandas as pd

import numba

import math

import sys

start_time_0=time.time()
im= str(sys.argv[1])#"B03_px-0545_py-1946"



#fn_in_ed="/data/homes/fcurvaia/Spheres_fit/in_edges_"+im+".npy"#"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/in_edges_"+im+".npy"
fn_out_ed="/data/homes/fcurvaia/Spheres_fit/out_edges_"+im+".npy"#"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/out_edges_"+im+".npy"
fn_in_sp="/data/homes/fcurvaia/Spheres_fit/in_sphere_"+im+".npy"#"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/in_sphere_"+im+".npy"
fn_out_sp="/data/homes/fcurvaia/Spheres_fit/out_sphere_"+im+".npy"#"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/out_sphere_"+im+".npy"
origin=np.load("/data/homes/fcurvaia/Spheres_fit/out_sphere_origin_"+im+".npy") #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/out_sphere_origin_"+im+".npy"



#in_edges=np.load(fn_in_ed)
out_edges=np.load(fn_out_ed)
in_sphere=np.load(fn_in_sp)
out_sphere=np.load(fn_out_sp)

#in_edges=np.unravel_index(np.where(in_edges.ravel()==True),in_edges.shape)
#out_edges=np.unravel_index(np.where(out_edges.ravel()==True),out_edges.shape)
#in_sphere=np.unravel_index(np.where(in_sphere.ravel()==True),in_sphere.shape)
#out_sphere=np.unravel_index(np.where(out_sphere.ravel()==True),out_sphere.shape)

#in_edges=np.array(np.where(in_edges==True))
out_edges=np.array(np.where(out_edges==True))
in_sphere=np.array(np.where(in_sphere==True))
out_sphere=np.array(np.where(out_sphere==True))


#in_edges_corr=in_edges*np.array([1,0.65,0.65])[:,np.newaxis]
#out_edges_corr=out_edges*np.array([1,0.65,0.65])[:,np.newaxis]
#in_sphere_corr=in_sphere*np.array([1,0.65,0.65])[:,np.newaxis]
#out_sphere_corr=out_sphere*np.array([1,0.65,0.65])[:,np.newaxis]


out_ed=pd.DataFrame(out_edges.T, columns=["z", "y", "x"])
out_ed[["z_corr", "y_corr", "x_corr"]]=out_ed[["z","y", "x"]]-list(origin)[::-1]

out_ed[["y_corr", "x_corr"]]=out_ed[["y_corr", "x_corr"]]*[0.65, 0.65]




print("Read and load file --- %s seconds ---\n" % (time.time() - start_time_0))


start_time_1=time.time()
# EUCLEDIAN DISTANCE Source: https://stackoverflow.com/questions/49664523/how-can-i-speed-up-closest-point-comparison-using-cdist-or-tensorflow
@numba.njit('(int64[:,::1], int64[::1], int64[::1], int64[::1])', parallel=True, fastmath=True)
def pz_dist_surf(p_array, x_flat, y_flat, z_flat):
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


def appendSpherical_pd_ed(df): #from https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    xy = df["x_corr"]**2+df["y_corr"]**2
    df["r"] = np.sqrt(xy + df["z_corr"]**2)
    #df["theta"] = pi+np.arctan2(np.sqrt(xy), df["Centroid_z"])
    df["theta"]= np.arccos(df["z_corr"]/df["r"])
    #df["phi"] = np.arctan2(df["Centroid_y"], df["Centroid_x"])
    df["phi"] = np.sign(df["y_corr"])*np.arccos(df["x_corr"]/np.sqrt(xy))
    return df

out_ed=appendSpherical_pd_ed(out_ed)
n_bins=36
phi_bins=np.linspace(-math.pi, math.pi, n_bins)

labels = range(1, n_bins)
out_ed['phi_bin'] = pd.to_numeric(pd.cut(x = out_ed['phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))

theta_bins=np.linspace(np.min(out_ed.theta), np.max(out_ed.theta)+0.000000001, n_bins, endpoint=True)

out_ed['theta_bin'] = pd.to_numeric(pd.cut(x = out_ed['theta'], bins = theta_bins, labels = labels, include_lowest = True, right=False))

def sample_in_group(grp):
    if len(grp)==0:
        pass
    elif len(grp)<5:
        return(grp.sample(n=len(grp)))
    else:
        return(grp.sample(n=5))


out_ed_samp=out_ed.groupby(['phi_bin', 'theta_bin'],as_index=False).apply(sample_in_group) #,as_index=False


dist_to_in_sphere=np.empty((0,))
for i in range(0, out_ed_samp.shape[0],2000):
    if i+2000>out_ed_samp.shape[0]:
        j=out_ed_samp.shape[0]
    else:
        j=i+2000
    start_time_3=time.time()
    min_dists_in_i=np.sqrt(np.min(pz_dist_surf(np.ascontiguousarray(out_ed_samp[["z","y","x"]])[i:j], in_sphere[0], in_sphere[1], in_sphere[2]),axis=1))
    dist_to_in_sphere=np.append(dist_to_in_sphere, min_dists_in_i)
    print("Get minimal inner distances --- %s seconds ---\n" % (time.time() - start_time_3))
del min_dists_in_i
dist_to_in_sphere=np.stack(dist_to_in_sphere, axis=0)

dist_to_out_sphere=np.empty((0,))
for i in range(0, out_ed_samp.shape[0],2000):
    if i+2000>out_ed_samp.shape[0]:
        j=out_ed_samp.shape[0]
    else:
        j=i+2000
    start_time_3=time.time()
    min_dists_out_i=np.sqrt(np.min(pz_dist_surf(np.ascontiguousarray(out_ed_samp[["z","y","x"]])[i:j], out_sphere[0], out_sphere[1], out_sphere[2]),axis=1))
    dist_to_out_sphere=np.append(dist_to_out_sphere, min_dists_out_i)
    print("Get minimal outer distances --- %s seconds ---\n" % (time.time() - start_time_3))
del min_dists_out_i
dist_to_out_sphere=np.stack(dist_to_out_sphere, axis=0)



#dist_to_out_sphere=np.sqrt(np.min(pz_dist_surf(np.ascontiguousarray(out_sphere.T), np.array(out_ed_samp["x"]),np.array(out_ed_samp["y"]),np.array(out_ed_samp["z"])),axis=1))
#dist_to_in_sphere=np.sqrt(np.min(pz_dist_surf(np.ascontiguousarray(in_sphere.T), np.array(out_ed_samp["x"]),np.array(out_ed_samp["y"]),np.array(out_ed_samp["z"])),axis=1))
print("Compute distances --- %s seconds ---\n" % (time.time() - start_time_1))

start_time_2=time.time()
out_ed_samp["d_i"]=dist_to_in_sphere
out_ed_samp["d_o"]=dist_to_out_sphere
means_in=out_ed_samp.groupby(['phi_bin', 'theta_bin'])["d_i"].mean()
means_out=out_ed_samp.groupby(['phi_bin', 'theta_bin'])["d_o"].mean()
#out_ed_samp.groupby(['phi_bin', 'theta_bin'])["r"].get_group((1,9)).mean()
out_ed["d_i"]=np.nan
out_ed["d_o"]=np.nan
n_col=len(out_ed.columns)
for i in range(1, n_bins):
    for j in range(1, n_bins):
        if (i,j) in means_in.index:
            out_ed.out_ed.loc[(out_ed["phi_bin"]==i) & (out_ed["theta_bin"]==j), n_col-1]=means_in.loc[(i,j)]
            out_ed.out_ed.loc[(out_ed["phi_bin"]==i) & (out_ed["theta_bin"]==j), n_col]=means_out.loc[(i,j)]

out_ed.to_csv("/data/homes/fcurvaia/"+im+"_w_dist_to_sph.csv")

print("filling df --- %s seconds ---\n" % (time.time() - start_time_2))



