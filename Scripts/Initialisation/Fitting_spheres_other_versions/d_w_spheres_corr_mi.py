#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:31:03 2023

@author: floriancurvaia
"""
import numpy as np

import time

import pandas as pd

import numba

import sys

import math

start_time_0=time.time()

path_in='/data/homes/fcurvaia/features/'
path_out="/data/homes/fcurvaia/distances/"
im=str(sys.argv[1])
fn=path_in+im+".csv"



features=pd.read_csv(fn, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "structure", "pSmad2/3_2_Mean" ]) #




print("Read and load files --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()
origin=np.load("/data/homes/fcurvaia/Spheres_fit/out_sphere_origin_"+im+".npy")


def appendSpherical_pd_ed(df): #from https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    xy = df["x_corr"]**2+df["y_corr"]**2
    df["r"] = np.sqrt(xy + df["z_corr"]**2)
    #df["theta"] = pi+np.arctan2(np.sqrt(xy), df["Centroid_z"])
    df["theta"]= np.arccos(df["z_corr"]/df["r"])
    #df["phi"] = np.arctan2(df["Centroid_y"], df["Centroid_x"])
    df["phi"] = np.sign(df["y_corr"])*np.arccos(df["x_corr"]/np.sqrt(xy))
    return df
feat_filt=features[(features.structure=='cells')]
del features
print(np.max(feat_filt[["Centroid_x","Centroid_y", "Centroid_z"]]))
feat_filt[["Centroid_x","Centroid_y", "Centroid_z"]]=feat_filt[["Centroid_x","Centroid_y", "Centroid_z"]]/[0.65,0.65,1]
print(np.max(feat_filt[["Centroid_x","Centroid_y", "Centroid_z"]]))
feat_filt[["x_corr","y_corr", "z_corr"]]=feat_filt[["Centroid_x","Centroid_y", "Centroid_z"]]-list(origin)[::-1]
feat_filt[["x_corr","y_corr", "z_corr"]]=feat_filt[["x_corr","y_corr", "z_corr"]]*[0.65,0.65,1]

feat_filt=appendSpherical_pd_ed(feat_filt)





cell_pos=np.ascontiguousarray(np.array(feat_filt[[ "Centroid_z", "Centroid_y","Centroid_x"]]))

in_sphere=np.load("/data/homes/fcurvaia/Spheres_fit/in_sphere_"+im+".npy")
in_sphere=in_sphere[0:306]

ed_id_in=np.array(np.where(in_sphere==True))


print("Prepare data --- %s seconds ---\n" % (time.time() - start_time_1))


# EUCLEDIAN DISTANCE
@numba.njit('(float64[:,::1], float64[::1], float64[::1], float64[::1])', parallel=True, fastmath=True)
def pz_dist(p_array, z_flat, y_flat, x_flat):
    m = p_array.shape[0]
    n = x_flat.shape[0]
    d = np.empty(shape=(m, n), dtype=np.float64)
    for i in numba.prange(m):
        p1 = p_array[i, :]
        for j in range(n):
            _z = z_flat[j] - p1[0]
            _y = (y_flat[j] - p1[1])*0.65 #p1[1])*0.65
            _x = (x_flat[j] - p1[2])*0.65 #p1[2])*0.65
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
del in_sphere, min_dists_in_i
min_dists_in=np.stack(min_dists_in, axis=0)

out_sphere=np.load("/data/homes/fcurvaia/Spheres_fit/out_sphere_"+im+".npy")
out_sphere=out_sphere[0:306]
ed_id_out=np.array(np.where(out_sphere==True))
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
del out_sphere, min_dists_out_i
min_dists_out=np.stack(min_dists_out, axis=0)

feat_filt["dist_in"]=min_dists_in
feat_filt["dist_out"]=min_dists_out



fn_out_ed="/data/homes/fcurvaia/Spheres_fit/out_edges_"+im+".npy"#"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/out_edges_"+im+".npy"
fn_in_sp="/data/homes/fcurvaia/Spheres_fit/in_sphere_"+im+".npy"#"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/in_sphere_"+im+".npy"
fn_out_sp="/data/homes/fcurvaia/Spheres_fit/out_sphere_"+im+".npy"#"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/out_sphere_"+im+".npy"
#origin=np.load("/data/homes/fcurvaia/Spheres_fit/out_sphere_origin_"+im+".npy") #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/out_sphere_origin_"+im+".npy"


out_edges=np.load(fn_out_ed)
#in_sphere=np.load(fn_in_sp)
#out_sphere=np.load(fn_out_sp)


out_edges=np.array(np.where(out_edges==True))
in_sphere=ed_id_in
out_sphere=ed_id_out


out_ed=pd.DataFrame(out_edges.T, columns=["z", "y", "x"])
out_ed[["z_corr", "y_corr", "x_corr"]]=out_ed[["z","y", "x"]]-list(origin)[::-1]

out_ed[["y_corr", "x_corr"]]=out_ed[["y_corr", "x_corr"]]*[0.65, 0.65]


print("Read and load file --- %s seconds ---\n" % (time.time() - start_time_0))


start_time_1=time.time()
# EUCLEDIAN DISTANCE Source: https://stackoverflow.com/questions/49664523/how-can-i-speed-up-closest-point-comparison-using-cdist-or-tensorflow
@numba.njit('(int64[:,::1], int64[::1], int64[::1], int64[::1])', parallel=True, fastmath=True)
def pz_dist_surf(p_array, z_flat, y_flat, x_flat):
    m = p_array.shape[0]
    n = x_flat.shape[0]
    d = np.empty(shape=(m, n), dtype=np.float64)
    for i in numba.prange(m):
        p1 = p_array[i, :]
        for j in range(n):
            _z = z_flat[j] - p1[0]
            _y = (y_flat[j] - p1[1])*0.65 #p1[1])*0.65
            _x = (x_flat[j] - p1[2])*0.65 #p1[2])*0.65
            _d = _x**2 + _y**2 + _z**2
            d[i, j] = _d
    return d




out_ed=appendSpherical_pd_ed(out_ed)
n_bins=72
phi_bins=np.linspace(-math.pi, math.pi, n_bins)

labels = range(1, n_bins)
out_ed['phi_bin'] = pd.to_numeric(pd.cut(x = out_ed['phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))

theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)

out_ed['theta_bin'] = pd.to_numeric(pd.cut(x = out_ed['theta'], bins = theta_bins, labels = labels, include_lowest = True, right=False))

feat_filt['phi_bin'] = pd.to_numeric(pd.cut(x = feat_filt['phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
feat_filt['theta_bin'] = pd.to_numeric(pd.cut(x = feat_filt['theta'], bins = theta_bins, labels = labels, include_lowest = True, right=False))


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
out_ed_samp.to_csv(path_out+im+"_out_ed_samp.csv", sep=",")
means_in=out_ed_samp.groupby(['phi_bin', 'theta_bin'])["d_i"].mean()
means_out=out_ed_samp.groupby(['phi_bin', 'theta_bin'])["d_o"].mean()
means_in.to_csv(path_out+im+"_means_in.csv")
means_out.to_csv(path_out+im+"_means_out.csv")
#means_in=pd.DataFrame(means_in)
#means_out=pd.DataFrame(means_out)
#out_ed_samp.groupby(['phi_bin', 'theta_bin'])["r"].get_group((1,9)).mean()
feat_filt["d_i"]=0
feat_filt["d_o"]=0
print(means_in.index)
print(means_out.index)
arr_means_in=np.zeros((n_bins,n_bins))
arr_means_out=np.zeros((n_bins,n_bins))
n_col=len(out_ed.columns)
for i in range(1, n_bins):
    for j in range(1, n_bins):
        if (i,j) in means_in.index:
            #print("yes")
            feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "d_i"]=means_in.loc[(i,j)]#.values[0]#[(means_in["phi_bin"]==i) & (means_in["theta_bin"]==j)].values[0]
            feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "d_o"]=means_out.loc[(i,j)]#.values[0]#[(means_out["phi_bin"]==i) & (means_out["theta_bin"]==j)].values[0]
            arr_means_in[i,j]=means_in.loc[(i,j)]
            arr_means_out[i,j]=means_out.loc[(i,j)]
feat_filt=feat_filt.sort_values("dist_out")
idxs_cells=feat_filt.groupby(['phi_bin', 'theta_bin'])["d_i"].mean().index
for i in range(1, n_bins):
    for j in range(1, n_bins):
        if (i,j) in idxs_cells:
            sor_out=feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "dist_out"]
            sor_in=feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "dist_in"]
            feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "rank_out"]=list(range(1, len(sor_out)+1))
            feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "rank_out_r"]=[x / len(sor_out) for x in list(range(1, len(sor_out)+1))]
            feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "dist_rel_out"]=sor_out-np.min(sor_out)
            feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "dist_rel_out_r"]=feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "dist_rel_out"]/np.max(feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "dist_rel_out"])
            feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "dist_rel_in"]=sor_in-np.min(sor_in)
            feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "dist_rel_in_r"]=feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "dist_rel_in"]/np.max(feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "dist_rel_in"])


print("filling df --- %s seconds ---\n" % (time.time() - start_time_2))




start_time_3=time.time()
feat_filt["d_corr"]=np.abs(feat_filt["dist_out"]-feat_filt["d_o"])/(np.abs(feat_filt["dist_in"]-feat_filt["d_i"])+np.abs(feat_filt["dist_out"]-feat_filt["d_o"]))
feat_filt["d_sor_r"]=feat_filt.dist_rel_out/(feat_filt.dist_rel_out+feat_filt.dist_rel_in)
feat_filt["d_sor_r_2"]= feat_filt["d_sor_r"]*feat_filt["dist_out"]

print("getting d_corr --- %s seconds ---\n" % (time.time() - start_time_3))


feat_filt.to_csv(path_out+im+"_w_dist_edge_corr_mi.csv", sep=",")
np.save(path_out+im+"_arr_means_in", arr_means_in)
np.save(path_out+im+"_arr_means_out", arr_means_out)



