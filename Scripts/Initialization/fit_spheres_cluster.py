#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:45:12 2023

@author: floriancurvaia
"""


import h5py
import numpy as np
import math
import time
import scipy.ndimage as ndimage
import pandas as pd
import plotly.express as px
import itk
from abbott.neighborhood_matrix_parallel import weighted_anisotropic_touch_matrix #, neighborhood_matrix
from abbott import itk_image
from abbott.conversions import *
import numba
import argparse
from pathlib import Path
import random


start_time_0=time.time()

CLI=argparse.ArgumentParser()
CLI.add_argument('idx', type=int)
CLI.add_argument(
  "--flds",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)
CLI.add_argument(
  "--colnames",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)

args = CLI.parse_args()
print(args.flds)
print(args.colnames)

#Path('/data/homes/fcurvaia/features/')
path_in_dist=Path('/data/homes/fcurvaia/pre_aligned/features/')
path_out_dist="/data/homes/fcurvaia/pre_aligned/distances/"
#Path("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/")
#path_in=Path("/data/active/fcurvaia/Segmented_files/")#Path(' '.join(args.flds))#Path('/data/active/sshami/20220716_experiment3_aligned/')
path_in=Path(args.flds[0]) 
path_out="/data/homes/fcurvaia/pre_aligned/Spheres_fit/"
path_out_im="/data/homes/fcurvaia/pre_aligned/Images/D_in_microns/"


image=list(path_in.glob("*.h5*"))[args.idx].name

colnames=args.colnames
im=image.split(".")[0]

fn_im=path_in / (im+".h5")
fn_csv=path_in_dist / (im+".csv")



n_bins=30
phi_bins=np.linspace(-math.pi, math.pi, n_bins)
labels = range(1, n_bins)
theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)

#Source:https://jekel.me/2015/Least-Squares-Sphere-Fit/
def sphereFit(coords):
    #   Assemble the A matrix
    spX = np.squeeze(coords[2])
    spY = np.squeeze(coords[1])
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

def sphereFit_2(coords):
    #   Assemble the A matrix
    spX = np.squeeze(coords[2])*0.65
    spY = np.squeeze(coords[1])*0.65
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



def appendSpherical_pd_ed(df): #from https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    xy = df["x_corr"]**2+df["y_corr"]**2
    df["r"] = np.sqrt(xy + df["z_corr"]**2)
    df["theta"]= np.arccos(df["z_corr"]/df["r"])
    df["phi"] = np.sign(df["y_corr"])*np.arccos(df["x_corr"]/np.sqrt(xy))
    return df
    
@numba.njit('(float64[:,::1], int64[:,::1], float64[::1], float64[::1], float64[::1])', parallel=True, fastmath=True)
def dist_btw_cells(arr_to_fill, idxs, Cx, Cy, Cz):
    for i in numba.prange(idxs.shape[1]):
        id_a=idxs[0, i]
        id_b=idxs[1, i]
        d=math.sqrt((Cx[id_a]-Cx[id_b])**2 + (Cy[id_a]-Cy[id_b])**2 + (Cz[id_a]-Cz[id_b])**2)
        arr_to_fill[id_a, id_b]=d
         
    return arr_to_fill

#Adapted from source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays
def find_edges(filename_im):
    with h5py.File(fn_im) as f:
        seg_emb=np.array(f["lbl_embryo"])
    mask=np.copy(seg_emb)
    seg_emb[~mask] = 0
    struct = ndimage.generate_binary_structure(3, 3)
    erode = ndimage.binary_erosion(mask, struct)
    edges = mask ^ erode
    return edges

def get_middle_sphere(edges, n_points=500):
    rand_id_z, rand_id_x, rand_id_y=np.unravel_index(
        random.sample(
            np.where(edges.ravel()==True)[0].tolist(),
            k=n_points), 
        edges.shape)
    indices=(rand_id_z, rand_id_y, rand_id_x)
    r, x0, y0, z0 = sphereFit(indices)
    z, y, x = np.ogrid[0:len(edges), 0:1000, 0:1000]
    sphere = np.add(np.add(np.square(x-x0), np.square(y-y0)), np.square(z-z0)) <=r**2
    return sphere

def get_outer_sphere(out_edges, n_points=5000):
    num_to_sample_slice=math.floor(n_points/len(out_edges))
    idx_x=np.empty((0,0), dtype="int64")
    idx_y=np.empty((0,0), dtype="int64")
    idx_z=np.empty((0,0), dtype="int64")
    z_slices_sums=np.sum(np.sum(out_edges, 2),1)
    for z, tot_z in zip(range(len(out_edges)), z_slices_sums):
        if tot_z==0:
            pass
        else:
            n_samp=min(tot_z, num_to_sample_slice)
            rand_id_x, rand_id_y=np.unravel_index(
                random.sample(
                    np.where(out_edges[z].ravel()==True)[0].tolist(),
                    k=n_samp), 
                out_edges[z].shape)
            id_z=np.full((1, n_samp), z)
            idx_x=np.concatenate((idx_x, rand_id_x), axis=None)
            idx_y=np.concatenate((idx_y, rand_id_y), axis=None)
            idx_z=np.concatenate((idx_z, id_z), axis=None)
    
    indices=(idx_z, idx_y, idx_x)
    r, x0, y0, z0 = sphereFit_2(indices)
    #z, y, x = np.ogrid[0:len(out_edges), 0:1000, 0:1000]
    #out_sphere = np.add(np.add(np.square(x-x0_2), np.square(y-y0_2)), np.square(z-z0_2)) <=r_2**2
    return(r, x0, y0, z0)# return(out_sphere, (r_2, x0_2, y0_2, z0_2))



def make_label_map(seg_cells):
    seg_cells=to_itk(seg_cells)
    seg_cells=to_labelmap(seg_cells)
    return seg_cells

def get_distance_matrix(fn_im, feat_filt):
    with h5py.File(fn_im) as f:
        seg_cells=np.array(f["lbl_cells"])
    seg_cells=make_label_map(seg_cells)
    adj_mat = weighted_anisotropic_touch_matrix(itk.GetArrayFromImage(itk_image.to_labelimage(seg_cells)).astype(np.int32))
    adj_mat=np.delete(adj_mat, 0, 0)
    adj_mat=np.delete(adj_mat, 0, 1)
    np.fill_diagonal(adj_mat, 0)
    adj_mat=adj_mat.astype("bool")
    to_rm=list(set(range(1, len(adj_mat)+1))-set(feat_filt.Label))
    to_rm=[i-1 for i in to_rm]
    to_rm.sort()
    adj_mat=np.delete(adj_mat, to_rm, 0)
    adj_mat=np.delete(adj_mat, to_rm, 1)
    dist_mat=np.zeros((adj_mat.shape))
    idxs_neigh=np.array(np.where(adj_mat==True))
    dist_mat=dist_btw_cells(dist_mat, idxs_neigh, feat_filt.Centroid_x.to_numpy(), feat_filt.Centroid_y.to_numpy(), feat_filt.Centroid_z.to_numpy())
    return(adj_mat, dist_mat)

def split_features(features):
    feat_nuc=features[(features.structure=='nucleiRaw')]
    feat_cyto=features[(features.structure=='cyto')]
    feat_filt=features[(features.structure=='cells')]
    labs_all=set(feat_nuc.Label).intersection(set(feat_filt.Label)).intersection(set(feat_cyto.Label))
    feat_cyto=feat_cyto.loc[feat_cyto["Label"].isin(labs_all)]
    feat_cyto.reset_index(drop=True, inplace=True)
    feat_nuc=feat_nuc.loc[feat_nuc["Label"].isin(labs_all)]
    feat_nuc.reset_index(drop=True, inplace=True)
    feat_filt=feat_filt.loc[feat_filt["Label"].isin(labs_all)]
    feat_filt.reset_index(drop=True, inplace=True)
    return(feat_filt, feat_cyto, feat_nuc)


            
print("load files and argument --- %s seconds ---\n" % (time.time() - start_time_0))



start_time_1=time.time()

edges=find_edges(fn_im)


print("Finding edges --- %s seconds ---\n" % (time.time() - start_time_1))

start_time_3=time.time()
out_edges=np.copy(edges)
out_edges[get_middle_sphere(edges)]=False
print("get middle sphere--- %s seconds ---\n" % (time.time() - start_time_3))


start_time_4=time.time()
r, x0, y0, z0 = get_outer_sphere(out_edges)
print("get outer sphere--- %s seconds ---\n" % (time.time() - start_time_4))


origin=np.array([x0[0], y0[0], -1*z0[0]])
print(origin)


start_time_4=time.time() 
features=pd.read_csv(fn_csv, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "structure", "NTouchingNeighbors", "PhysicalSize"]+colnames)#, "pSmad1/5_4_Mean", "betaCatenin_0_Mean", "MapK_2_Mean" ]) #"pSmad1/5_4_Mean"
features["Centroid_z"]*=-1
features=features.sort_values("Label")
features[["x_corr","y_corr", "z_corr"]]=features[["Centroid_x","Centroid_y", "Centroid_z"]]-[origin[0],origin[1],origin[2]]
features=appendSpherical_pd_ed(features)
features[["x_corr","y_corr", "z_corr"]]=features[["x_corr","y_corr", "z_corr"]]/[0.65,0.65,1]

for col in colnames:
    features=features.drop(features[features[col]==0].index)
    features=features.drop(features[features[col]<0.5].index)
    

feat_filt, feat_cyto, feat_nuc = split_features(features)


feat_filt["CytoSize"]=feat_cyto["PhysicalSize"]
feat_filt["NucSize"]=feat_nuc["PhysicalSize"]

start_time_3=time.time() 
adj_mat, dist_mat=get_distance_matrix(fn_im, feat_filt)
print("get adjacency and distance matrix --- %s seconds ---\n" % (time.time() - start_time_3))



r_neigh=[max(i,j) for i,j in zip(np.max(dist_mat, axis=0), np.max(dist_mat, axis=1))]
feat_filt["r_neigh"]=r_neigh

np.save(path_out+im+"_adj_mat", adj_mat)
np.save(path_out+im+"_dist_mat", dist_mat)
dist_mat[dist_mat==0]=np.nan

r_neigh_mean=np.nanmean(dist_mat, axis=1)
r_neigh_mean[r_neigh_mean==np.nan]=0
feat_filt["r_neigh_mean"]=r_neigh_mean

for col in colnames:
    stain=col.split("_")[0]
    ratio=stain+"_nc_ratio"
    feat_filt[ratio]=(feat_nuc[col].to_numpy()/feat_cyto[col].to_numpy())
    feat_filt.replace([np.inf, -np.inf], np.nan, inplace=True)
    #feat_filt[ratio].fillna(feat_nuc[col], inplace=True)
    feat_filt[stain+"_nuc"]=feat_nuc[col]
    feat_filt[stain+"_cyto"]=feat_cyto[col]
    


feat_filt["dist_out"]=np.abs(np.sqrt((origin[0]-feat_filt.Centroid_x)**2+(origin[1]-feat_filt.Centroid_y)**2+(origin[2]-feat_filt.Centroid_z)**2)-r)#*0.65

del features

print("Generate new csv --- %s seconds ---\n" % (time.time() - start_time_4))
np.save(path_out+"out_sphere_origin_"+im, origin )
np.save(path_out+"out_sphere_radius_"+im, r )

feat_filt.to_csv(path_out_dist+im+"_w_dist_sph_simp.csv", sep=",", index=False)


fig3 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="dist_out", opacity=1, color_continuous_scale="turbo")
fig3.update_layout(margin=dict(l=0, r=0, b=0, t=0), coloraxis_colorbar_title_text = 'Distance')
fig3.write_html(path_out_im+"dist_out_simp_"+im+".html")

fig4 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color='phi', opacity=1, color_continuous_scale="turbo")
fig4.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig4.write_html(path_out_im+"phi_"+im+".html")

fig5 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color='theta', opacity=1, color_continuous_scale="turbo")
fig5.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig5.write_html(path_out_im+"theta_"+im+".html")


for col in colnames:
    c=col.split("_")
    stain=c[0]
    stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
    
    fig_col_1 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color=stain+'_nc_ratio', opacity=1, color_continuous_scale="turbo", range_color=[feat_filt[stain+"_nc_ratio"].quantile(0.05), feat_filt[stain+"_nc_ratio"].quantile(0.95)])
    fig_col_1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig_col_1.write_html(path_out_im+stain_clean+"_nc_ratio_"+im+".html")
    
    fig_col_2 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color=stain+"_nuc", opacity=1, color_continuous_scale="turbo", range_color=[feat_filt[stain+"_nuc"].quantile(0.05), feat_filt[stain+"_nuc"].quantile(0.95)])
    fig_col_2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig_col_2.write_html(path_out_im+stain_clean+"_nuc_mean_"+im+".html")
    
    fig_col_3 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color=col, opacity=1, color_continuous_scale="turbo", range_color=[feat_filt[col].quantile(0.05), feat_filt[col].quantile(0.95)])
    fig_col_3.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig_col_3.write_html(path_out_im+stain_clean+"_"+im+".html")
    
    fig_col_4 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color=stain+"_cyto", opacity=1, color_continuous_scale="turbo", range_color=[feat_filt[stain+"_cyto"].quantile(0.05), feat_filt[stain+"_cyto"].quantile(0.95)])
    fig_col_4.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig_col_4.write_html(path_out_im+stain_clean+"_cyto_mean_"+im+".html")
   

fig10 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="r_neigh", opacity=1, color_continuous_scale="turbo")
fig10.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig10.write_html(path_out_im+"r_neigh_"+im+".html")

fig11 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="r_neigh_mean", opacity=1, color_continuous_scale="turbo")
fig11.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig11.write_html(path_out_im+"r_neigh_mean_"+im+".html")
