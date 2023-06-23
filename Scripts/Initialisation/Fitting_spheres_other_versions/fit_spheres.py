#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:06:50 2023

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


start_time_0=time.time()

CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--images",  # name on the CLI - drop the `--` for positional/required parameters
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
path_in_dist='/data/homes/fcurvaia/features/' 
path_out_dist="/data/homes/fcurvaia/distances/" 
path_in='/data/active/sshami/20220716_experiment3_aligned/'
path_out="/data/homes/fcurvaia/Spheres_fit/"
path_out_im="/data/homes/fcurvaia/Images/D_in_microns/"

wells=["D06", "B07", "C07", "D07"]
fld=Path(path_in)
files=[]
for w in wells:
    for f in fld.glob(w+"*"):
        name=f.name
        files.append(name.split(".")[0])

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
    C, residules, rank, singval = np.linalg.lstsq(A,f)

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
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2] 

def get_adjacency_matrix(
lbl_map: itk.LabelMap
) -> np.array:
    nm = weighted_anisotropic_touch_matrix(
        itk.GetArrayFromImage(itk_image.to_labelimage(lbl_map)).astype(np.int32)
    )
    return(nm)

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


for im in files:#args.images:
    fn=path_in+im+".h5"
    fn_dist=path_in_dist+im+".csv"
    
    with h5py.File(fn) as f:
        seg_emb=np.array(f["lbl_embryo"])
        seg_cells=np.array(f["lbl_cells"])
            
    print("load files and argument --- %s seconds ---\n" % (time.time() - start_time_0))
    
    z_width_low=math.floor(seg_emb.shape[0]*0.05)
    z_width_high=seg_emb.shape[0]-math.floor(seg_emb.shape[0]*0.1)
    
    
    start_time_1=time.time()
    mask=np.copy(seg_emb)
    seg_emb[~mask] = 0
    print("doing mask --- %s seconds ---\n" % (time.time() - start_time_1))
    
    #Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays
    start_time_2=time.time()
    struct = ndimage.generate_binary_structure(3, 3)
    erode = ndimage.binary_erosion(mask, struct)
    edges = mask ^ erode
    
    
    print("erosion --- %s seconds ---\n" % (time.time() - start_time_2))
    del mask, erode, struct, seg_emb
    

    
    
    start_time_3=time.time()
    tu=np.unravel_index(np.where(edges.ravel()==True),edges.shape)
    idx=np.random.randint(len(np.squeeze(tu[0])), size=500)
    tu_1=(tu[0][:,idx],tu[1][:,idx], tu[2][:,idx])
    r, x0, y0, z0 = sphereFit(tu_1)
    print("fitting middle sphere--- %s seconds ---\n" % (time.time() - start_time_3))
    z, y, x = np.ogrid[0:len(edges), 0:1000, 0:1000]
    mid_sphere = np.add(np.add(np.square(x-x0), np.square(y-y0)), np.square(z-z0)) <=r**2
    print("get middle sphere--- %s seconds ---\n" % (time.time() - start_time_3))
    del tu, idx, tu_1, z, y, x
    
    
    
    mask=np.copy(mid_sphere)
    mid_sphere[~mask] = 0
    
    #Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays
    
    struct = ndimage.generate_binary_structure(3, 3)
    erode = ndimage.binary_erosion(mask, struct)
    mid_sphere_ed = mask ^ erode
    
    start_time_4=time.time()
    
    out_edges=np.copy(edges)
    out_edges[mid_sphere]=False
    del mid_sphere, mask, erode, struct
    
    num_to_sample_slice=math.floor(5000/len(out_edges))
    idx_x_2=np.empty((0,0), dtype="int64")
    idx_y_2=np.empty((0,0), dtype="int64")
    idx_z_2=np.empty((0,0), dtype="int64")
    for sli in range(1,len(edges)-1):
        tot_sli=np.sum(out_edges[sli])
        if num_to_sample_slice>tot_sli:
            if tot_sli==0:
                pass
            elif tot_sli==1:
                tu_sli=np.unravel_index(np.where(out_edges[sli].ravel()==True),out_edges[sli].shape)
                tu_sli_samp=(tu_sli[1][:,0], tu_sli[0][:,0])
                tu_sli_z = np.full((1, tot_sli), sli)
                idx_x_2=np.concatenate((idx_x_2,tu_sli_samp[0]), axis=None)
                idx_y_2=np.concatenate((idx_y_2,tu_sli_samp[1]), axis=None)
                idx_z_2=np.concatenate((idx_z_2, tu_sli_z), axis=None)
            else:
                tu_sli=np.unravel_index(np.where(out_edges[sli].ravel()==True),out_edges[sli].shape)
                to_samp=np.random.randint(len(np.squeeze(tu_sli[0])), size=tot_sli)
                tu_sli_samp=(tu_sli[1][:,to_samp], tu_sli[0][:,to_samp])
                tu_sli_z = np.full((1, tot_sli), sli)
                idx_x_2=np.concatenate((idx_x_2,tu_sli_samp[0]), axis=None)
                idx_y_2=np.concatenate((idx_y_2,tu_sli_samp[1]), axis=None)
                idx_z_2=np.concatenate((idx_z_2, tu_sli_z), axis=None)
        else:
            tu_sli=np.unravel_index(np.where(out_edges[sli].ravel()==True),out_edges[sli].shape)
            to_samp=np.random.randint(len(np.squeeze(tu_sli[0])), size=num_to_sample_slice)
            tu_sli_samp=(tu_sli[1][:,to_samp], tu_sli[0][:,to_samp])
            tu_sli_z = np.full((1, num_to_sample_slice), sli)
            idx_x_2=np.concatenate((idx_x_2,tu_sli_samp[0]), axis=None)
            idx_y_2=np.concatenate((idx_y_2,tu_sli_samp[1]), axis=None)
            idx_z_2=np.concatenate((idx_z_2, tu_sli_z), axis=None)
        
    tu=(idx_z_2, idx_y_2, idx_x_2)
    
    
    
    r_2, x0_2, y0_2, z0_2 = sphereFit_2(tu)
    print("fitting outer sphere--- %s seconds ---\n" % (time.time() - start_time_4))
    z, y, x = np.ogrid[0:len(out_edges), 0:1000, 0:1000]
    out_sphere = np.add(np.add(np.square(x-x0_2), np.square(y-y0_2)), np.square(z-z0_2)) <=r_2**2
    print("get outer sphere--- %s seconds ---\n" % (time.time() - start_time_4))
    
    
    print(r, x0, y0, z0)
    print(r_2, x0_2, y0_2, z0_2)
    print(x0-x0_2,y0-y0_2,z0-z0_2)
    
    origin=np.array([x0_2[0], y0_2[0], -1*z0_2[0]])
    
        
    start_time_1=time.time() 
    seg_cells=to_itk(seg_cells)
    print("to_itk --- %s seconds ---\n" % (time.time() - start_time_1))
    
    start_time_2=time.time() 
    seg_cells=to_labelmap(seg_cells)
    print("to_labelmap --- %s seconds ---\n" % (time.time() - start_time_2))
    
    
    start_time_3=time.time() 
    
    adj_mat=get_adjacency_matrix(seg_cells)
    print("get adjacency matrix --- %s seconds ---\n" % (time.time() - start_time_3))
    start_time_4=time.time() 
    adj_mat=np.delete(adj_mat, 0, 0)
    adj_mat=np.delete(adj_mat, 0, 1)
    np.fill_diagonal(adj_mat, 0)
    adj_mat=adj_mat.astype("bool")
    
    features=pd.read_csv(fn_dist, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "structure", "NTouchingNeighbors", "PhysicalSize", "EquivalentEllipsoidDiameter_x", "EquivalentEllipsoidDiameter_y", "EquivalentEllipsoidDiameter_z"]+args.colnames)#, "pSmad1/5_4_Mean", "betaCatenin_0_Mean", "MapK_2_Mean" ]) #"pSmad1/5_4_Mean"
    features["Centroid_z"]*=-1
    features=features.sort_values("Label")
    features[["x_corr","y_corr", "z_corr"]]=features[["Centroid_x","Centroid_y", "Centroid_z"]]-[origin[0],origin[1],origin[2]]
    features=appendSpherical_pd_ed(features)
    features[["x_corr","y_corr", "z_corr"]]=features[["x_corr","y_corr", "z_corr"]]/[0.65,0.65,1]
    
    for col in args.colnames:
        features=features.drop(features[features[col]==0].index)
        features=features.drop(features[features[col]<0.5].index)
        
    
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
    
    
    feat_filt["CytoSize"]=feat_cyto["PhysicalSize"]
    feat_filt["NucSize"]=feat_nuc["PhysicalSize"]
    
    to_rm=list(set(range(1, len(adj_mat)+1))-set(feat_filt.Label))
    to_rm=[i-1 for i in to_rm]
    to_rm.sort()
    
    adj_mat=np.delete(adj_mat, to_rm, 0)
    adj_mat=np.delete(adj_mat, to_rm, 1)
    
    dist_mat=np.zeros((adj_mat.shape))
    idxs_neigh=np.array(np.where(adj_mat==True))
    
    
    dist_mat=dist_btw_cells(dist_mat, idxs_neigh, feat_filt.Centroid_x.to_numpy(), feat_filt.Centroid_y.to_numpy(), feat_filt.Centroid_z.to_numpy())
    
    r_neigh=[max(i,j) for i,j in zip(np.max(dist_mat, axis=0), np.max(dist_mat, axis=1))]
    feat_filt["r_neigh"]=r_neigh
    
    np.save(path_out+im+"_adj_mat", adj_mat)
    np.save(path_out+im+"_dist_mat", dist_mat)
    dist_mat[dist_mat==0]=np.nan
    
    r_neigh_mean=np.nanmean(dist_mat, axis=1)
    r_neigh_mean[r_neigh_mean==np.nan]=0
    feat_filt["r_neigh_mean"]=r_neigh_mean
    
    print("get distance matrix --- %s seconds ---\n" % (time.time() - start_time_4))
    
    for col in args.colnames:
        stain=col.split("_")[0]
        ratio=stain+"_nc_ratio"
        feat_filt[ratio]=(feat_nuc[col].to_numpy()/feat_cyto[col].to_numpy())
        feat_filt.replace([np.inf, -np.inf], np.nan, inplace=True)
        #feat_filt[ratio].fillna(feat_nuc[col], inplace=True)
        feat_filt[stain+"_nuc"]=feat_nuc[col]
        feat_filt[stain+"_cyto"]=feat_cyto[col]
        feat_filt[stain+"_nc_diff"]=feat_nuc[col].to_numpy()-feat_cyto[col].to_numpy()
        feat_filt[stain+"_nc_diff_ratio"]=feat_nuc[col].to_numpy()/(feat_nuc[col].to_numpy()+feat_cyto[col].to_numpy())
        
        
        print(" ".join(["Stain: ", str(col), "Min filt: ", str(np.min(feat_filt[col])), "Min nuc: ", str(np.min(feat_nuc[col])), "Min cyto: ", str(np.min(feat_cyto[col]))]))
        print(" ".join(["Stain: ", str(col), "Max filt: ", str(np.max(feat_filt[col])), "Max nuc: ", str(np.max(feat_nuc[col])), "Max cyto: ", str(np.max(feat_cyto[col]))]))
    

    feat_filt["dist_out"]=np.abs(np.sqrt((x0_2-feat_filt.Centroid_x)**2+(y0_2-feat_filt.Centroid_y)**2+(-1*z0_2-feat_filt.Centroid_z)**2)-r_2)#*0.65
    feat_filt["dist_out_r"]=feat_filt["dist_out"]/r_2
    del features
    
    mask=np.copy(out_sphere)
    out_sphere[~mask] = 0
    
    #Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays
    
    struct = ndimage.generate_binary_structure(3, 3)
    erode = ndimage.binary_erosion(mask, struct)
    out_sphere_ed = mask ^ erode
    
    del mask, erode, struct
    
    np.save(path_out+"out_sphere_origin_"+im, origin )
    np.save(path_out+"out_sphere_radius_"+im, r_2 )
    
    
    del idx_z_2, idx_x_2, idx_y_2, tu, z, y, x, out_sphere
    
    feat_filt["xy_ratio"]= feat_filt.EquivalentEllipsoidDiameter_x/feat_filt.EquivalentEllipsoidDiameter_y
    feat_filt["xz_ratio"]= feat_filt.EquivalentEllipsoidDiameter_x/feat_filt.EquivalentEllipsoidDiameter_z
    feat_filt["yz_ratio"]= feat_filt.EquivalentEllipsoidDiameter_y/feat_filt.EquivalentEllipsoidDiameter_z
    feat_filt["aspect_ratio_max"]=feat_filt[["xy_ratio","xz_ratio","yz_ratio"]].max(axis=1)
    
    
    feat_filt['phi_bin'] = pd.to_numeric(pd.cut(x = feat_filt['phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
    feat_filt['theta_bin'] = pd.to_numeric(pd.cut(x = feat_filt['theta'], bins = theta_bins, labels = labels, include_lowest = True, right=False))
    
    
    feat_filt.to_csv(path_out_dist+im+"_w_dist_sph_simp.csv", sep=",", index=False)
    
    

    fig1 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color='phi_bin', opacity=1, color_continuous_scale="turbo")
    fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig1.write_html(path_out_im+"phi_bin_"+im+"_"+str(n_bins)+".html")
    
    fig2 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color='theta_bin', opacity=1, color_continuous_scale="turbo")
    fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig2.write_html(path_out_im+"theta_bin_"+im+"_"+str(n_bins)+".html")
    
    fig3 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="dist_out_r", opacity=1, color_continuous_scale="turbo")
    fig3.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig3.write_html(path_out_im+"dist_out_simp_r_"+im+".html")
    
    fig4 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color='phi', opacity=1, color_continuous_scale="turbo")
    fig4.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig4.write_html(path_out_im+"phi_"+im+".html")
    
    fig5 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color='theta', opacity=1, color_continuous_scale="turbo")
    fig5.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig5.write_html(path_out_im+"theta_"+im+".html")
    
    
    for col in args.colnames:
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
        
        fig_col_5 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color=stain+'_nc_diff', opacity=1, color_continuous_scale="turbo", range_color=[feat_filt[stain+"_nc_diff"].quantile(0.05), feat_filt[stain+"_nc_diff"].quantile(0.95)])
        fig_col_5.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig_col_5.write_html(path_out_im+stain_clean+"_nc_diff_"+im+".html")
        
        fig_col_6 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color=stain+'_nc_diff_ratio', opacity=1, color_continuous_scale="turbo", range_color=[feat_filt[stain+"_nc_diff_ratio"].quantile(0.05), feat_filt[stain+"_nc_diff_ratio"].quantile(0.95)])
        fig_col_6.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig_col_6.write_html(path_out_im+stain_clean+"_nc_diff_ratio_"+im+".html")
    
    fig8 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="dist_out", opacity=1, color_continuous_scale="turbo")
    fig8.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig8.write_html(path_out_im+"dist_out_simp_"+im+".html")
    
    fig10 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="r_neigh", opacity=1, color_continuous_scale="turbo")
    fig10.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig10.write_html(path_out_im+"r_neigh_"+im+".html")
    
    fig11 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="r_neigh_mean", opacity=1, color_continuous_scale="turbo")
    fig11.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig11.write_html(path_out_im+"r_neigh_mean_"+im+".html")
   