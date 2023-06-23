#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:06:25 2023

@author: floriancurvaia
"""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from math import pi


#B02_px-0280_py+1419, B03_px-0545_py-1946, B04_px+0114_py-1436, B05_px+1522_py-1087, C05_px+0198_py+1683, 
#D05_px-0738_py-1248, B06_px-0493_py+0146, D06_px-1055_py-0118, B07_px+1257_py-0474, B08_px+1076_py-0189
#C04_px-0816_py-1668 INSTEAD OF B04




images=["B02_px-0280_py+1419", "B03_px-0545_py-1946", "C04_px-0816_py-1668", "B05_px+1522_py-1087", "C05_px+0198_py+1683", 
        "D05_px-0738_py-1248", "B06_px-0493_py+0146", "D06_px-1055_py-0118", "B07_px+1257_py-0474", "B08_px+1076_py-0189"]
stains=['pSmad1/5_nc_ratio', 'betaCatenin_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio', 'pSmad2/3_nc_ratio']


time_emb={"B02":3.3, "C02":3.3, "D02":3.3, "B03":3.7, "C03":3.7, "D03":3.7, "B04":4.3, "C04":4.3, "D04":4.3,
          "B05":4.7, "C05":5.3, "D05":5.7, "B06":5.7, "C06":5.7, "D06":6.0, "B07":6.0, "C07":6.0, "D07":6.0,
          "B08":7.0, "C08":7.0, "D08":7.0}

path_in_adj="/data/homes/fcurvaia/Spheres_fit/"

path_out_im="/data/homes/fcurvaia/Images/Theta_scatter/"

all_df=[]
for im in images:
    fn1="/data/homes/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv"
    #feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "x_corr", "y_corr", "z_corr", "structure", "r_neigh", "r_neigh_mean", "NTouchingNeighbors","PhysicalSize", "theta","phi", "phi_bin", "theta_bin", "dist_out", "phi_bin_new", "phi_bin_cur", "new_phi", "cur_phi"]+stains)
    feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["Label", "r_neigh", "NTouchingNeighbors","PhysicalSize", "theta", "dist_out", "cur_phi"]+stains)
    well=im.split("_")[0]
    feat_filt["hpf"]=time_emb[well]
    feat_filt["emb"]=well
    feat_filt.cur_phi=feat_filt.cur_phi.abs()
    all_df.append(feat_filt)
    
feat_filt_all=pd.concat(all_df)

for stain in stains:
    c=stain.split("_")
    stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
    
    plt.style.use('dark_background')
    f1, axs1 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.25]}, figsize=(60, 30))
    fon_size=15
    #space_size=38
    plt.rcParams.update({'font.size': fon_size}) 
    plt.subplots_adjust(hspace=0.125)
    plt.subplots_adjust(wspace=0.125)
    f1.set_dpi(300)
    f2, axs2 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.25]}, figsize=(60, 30))
    plt.subplots_adjust(hspace=0.125)
    plt.subplots_adjust(wspace=0.125)
    f2.set_dpi(300)
    for im, ax, ax1 in zip(images, axs1.flatten(), axs2.flatten()):
        
        #fn1="/data/homes/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv"
        
        #fn1="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"+im+"_w_dist_sph_simp.csv"
        #feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["Label", "r_neigh", "NTouchingNeighbors","PhysicalSize", "theta", "dist_out", "cur_phi"]+stains)
        #feat_filt.cur_phi=np.abs(feat_filt.cur_phi)
        well=im.split("_")[0]
        hpf=time_emb[well]
        feat_filt=feat_filt_all.loc[feat_filt_all.emb==well]
        adj_mat=np.load(path_in_adj+im+"_adj_mat.npy")
        #bcat_neigh=np.ma.median(np.ma.masked_equal(np.multiply(adj_mat, feat_filt.betaCatenin_nuc.to_numpy()),0), axis=1).data
        stain_neigh=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 1, np.multiply(adj_mat, feat_filt[stain].to_numpy()))
        
        if stain=="betaCatenin_nuc":
            feat_filt[stain+"_neigh"]=(feat_filt[stain]-stain_neigh)/feat_filt[stain]
            feat_filt=feat_filt.loc[feat_filt.betaCatenin_nuc>=1]
        else:
            feat_filt[stain+"_neigh"]=feat_filt[stain]-stain_neigh
        
        y_min=feat_filt_all[stain].quantile(0.01)
        y_max=feat_filt_all[stain].quantile(0.99)
        ax_=ax.scatter(
            x=feat_filt.theta, y=feat_filt[stain],
            c=feat_filt["cur_phi"], vmin=0, vmax=pi, marker="o", s=16, cmap="jet", alpha=0.5)
        ax.set_title(im+" "+str(hpf)+" hpf")
        if well=="C05" or well=="B08":
            f1.colorbar(ax_, ax=ax, label="cur_phi")
        
        ax.set_xlabel("theta")
        ax.set_ylabel(stain)
        ax.set_ylim(y_min,y_max)
        
        ax_1=ax1.scatter(
            x=feat_filt.theta, y=feat_filt[stain+"_neigh"],
            c=feat_filt["cur_phi"], vmin=0, vmax=pi, marker="o", s=16, cmap="jet", alpha=0.5)
        ax1.set_title(im+" "+str(hpf)+" hpf")
        if well=="C05" or well=="B08":
            f2.colorbar(ax_1, ax=ax1, label="cur_phi")
        
        ax1.set_xlabel("theta")
        ax1.set_ylabel(stain+"_shift")
        ax1.set_ylim(-1,1.5)
        
    f1.savefig(path_out_im+stain_clean+"_scatter.png", bbox_inches='tight', dpi=300)
    f2.savefig(path_out_im+stain_clean+"_scatter_median.png", bbox_inches='tight', dpi=300)
    plt.close()





