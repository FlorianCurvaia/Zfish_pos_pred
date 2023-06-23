#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:30:41 2023

@author: floriancurvaia
"""


import numpy as np

import time

import pandas as pd

import math

import matplotlib.pyplot as plt

import argparse

import plotly.express as px

import seaborn as sns

from scipy.stats import zscore



images=["B02_px-0280_py+1419"]#, "B03_px-0545_py-1946", "C04_px-0816_py-1668", "B05_px+1522_py-1087", "C05_px+0198_py+1683",
        #"D05_px-0738_py-1248", "B06_px-0493_py+0146", "D06_px-1055_py-0118", "B07_px+1257_py-0474", "B08_px+1076_py-0189"]
#C04_px-0816_py-1668 INSTEAD OF B04_px+0114_py-1436

start_time_0=time.time()
"""
CLI=argparse.ArgumentParser()

CLI.add_argument(
  "--colnames",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)

args = CLI.parse_args()
path_in_dist="/data/homes/fcurvaia/distances/"

path_out_im="/data/homes/fcurvaia/Images/Heatmaps/Sing_cells/"

print(args.colnames)
stains=[]
dyes=[]
for col in args.colnames:
    stain=col.split("_")[0]
    if stain=="betaCatenin":
        nuc=stain+"_nuc"
        stains.append(nuc)
        dyes.append(nuc)
    else:
        ratio=stain+"_nc_ratio"
        stains.append(ratio)
        dyes.append(ratio)
    #cyto=stain+"_cyto"
    #diff=stain+"_nc_diff"
    #diff_ratio=stain+"_nc_diff_ratio"
    #Path(path_out_im+"").mkdir(parents=True, exist_ok=True)
    #stains.append(col)
    
    #stains.append(cyto)
    #stains.append(diff)
    #stains.append(diff_ratio)

"""
#path_out_im="/data/homes/fcurvaia/Images/Heatmaps/Sing_cells/"
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Scatter_all/"
stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio', 'pSmad2/3_nc_ratio']
dyes=stains

print(stains)

time_emb={"B02":3.3, "C02":3.3, "D02":3.3, "B03":3.7, "C03":3.7, "D03":3.7, "B04":4.3, "C04":4.3, "D04":4.3,
          "B05":4.7, "C05":5.3, "D05":5.7, "B06":5.7, "C06":5.7, "D06":6.0, "B07":6.0, "C07":6.0, "D07":6.0,
          "B08":7.0, "C08":7.0, "D08":7.0}

prop=["cur_phi", "theta", "dist_out"]
all_df=[]
for im in images:
    #fn1=path_in_dist+im+"_w_dist_sph_simp.csv"
    fn1="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"+im+"_w_dist_sph_simp.csv"
    #fn1="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv"
    #feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "x_corr", "y_corr", "z_corr", "structure", "r_neigh", "r_neigh_mean", "NTouchingNeighbors","PhysicalSize", "theta","phi", "phi_bin", "theta_bin", "dist_out", "phi_bin_new", "phi_bin_cur", "new_phi", "cur_phi"]+stains)
    feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["theta","theta_bin", "dist_out",  "cur_phi"]+stains)
    well=im.split("_")[0]
    feat_filt["hpf"]=time_emb[well]
    feat_filt["emb"]=well
    feat_filt.cur_phi=feat_filt.cur_phi.abs()
    for col in dyes:
        feat_filt[col]=zscore(feat_filt[col], nan_policy="omit")
    for col in prop:
        feat_filt[col]=zscore(feat_filt[col], nan_policy="omit")
    all_df.append(feat_filt)
    
feat_filt=pd.concat(all_df)

#for col in dyes:
#    feat_filt[col]=zscore(feat_filt[col], nan_policy="omit")

print("load files --- %s seconds ---\n" % (time.time() - start_time_0))



start_time_1=time.time()
n_bins=30
for stain in stains:
    feat_filt=feat_filt.drop(feat_filt[feat_filt[stain]==0].index)

phi_bins=np.linspace(-math.pi, math.pi, n_bins)

theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)

## Phi-Theta heatmaps for 3 distance classes
#pSmad2/3_nc_ratio
dist_bins_ang=np.linspace(0, np.max(feat_filt.dist_out)+0.000000001, 4)

t_max=np.max(feat_filt.theta_bin)

n_bins_d=10

dist_bins=np.linspace(0, np.max(feat_filt.dist_out)+0.000000001, n_bins_d)

t_max=np.max(feat_filt.theta_bin)

theta_bins=theta_bins[:t_max]
theta_labs=360*theta_bins/(2*math.pi)
theta_labs=theta_labs#[:t_max]
phi_labs=360*phi_bins/(2*math.pi)
phi_labs=phi_labs
print("make umap --- %s seconds ---\n" % (time.time() - start_time_1))

start_time_2=time.time()
#path_out_im="/data/homes/fcurvaia/Images/UMAP/Common/W_time/"

#path_out_im="/data/homes/fcurvaia/Images/UMAP/Common/"

#path_out_im="/data/homes/fcurvaia/Images/UMAP/Common/PTD/"

#prop=["cur_phi", "theta", "dist_out", "r_neigh"]

prop+=["hpf"]

dyes+=prop

plt.style.use('dark_background')
l=len(dyes)
#w, h=figaspect(900/1440)

for im in images:
    well=im.split("_")[0]
    feat_filt_im=feat_filt.loc[feat_filt.emb==well]
    #comp1_im=feat_filt_im.comp_1
    #comp2_im=feat_filt_im.comp_2
    
    feat_filt_ot=feat_filt.loc[feat_filt.emb!=well]
    #comp1_ot=feat_filt_ot.comp_1
    #comp2_ot=feat_filt_ot.comp_2
    
    f1, axs1 = plt.subplots(2,4, gridspec_kw={'width_ratios': [1, 1, 1, 1]}, figsize=(60, 30))#
    plt.clf()
    fon_size=15
    space_size=38
    plt.rcParams.update({'font.size': fon_size}) 
    #sns.set_style("white",  {'figure.facecolor': 'white'})
    plt.subplots_adjust(hspace=0.125)
    plt.subplots_adjust(wspace=0.125)
    f1.set_dpi(300)
    for s, ax in zip(range(l), axs1.flatten()):
        col_map="jet"
        stain=dyes[s]
        #Phi Theta heatmaps
        """
        if stain=="cur_phi":
            col_map="twilight_shifted"
        else:
            col_map="jet"
        """
        #stain_bins=np.linspace(0, np.max(feat_filt[stain]), 20)
        c=stain.split("_")
        stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
        if stain in prop[0:2]:
            vmin=feat_filt_im[stain].min()
            vmax=feat_filt_im[stain].max()
            cbar_lab=stain
        elif stain in prop:
            vmin=feat_filt_im[stain].quantile(0.01)
            vmax=feat_filt_im[stain].quantile(0.99)
            cbar_lab=stain
        else:
            vmin=feat_filt_im[stain].quantile(0.01)
            vmax=feat_filt_im[stain].quantile(0.99)
            cbar_lab=stain+" intensity"
        
        """
        ax.scatter(
            x=comp1_ot, y=comp2_ot, #data=pacmap_proj
            marker="o", s=16, color="dimgrey", alpha=0.25)
        """
        ax_=ax.scatter(data=feat_filt_ot,
            x="theta", y="cur_phi",
            marker="o", s=16, color="dimgrey", alpha=0.25)
            
        ax.set_title(stain_clean)
        g = sns.jointplot(data=feat_filt_im, x="theta", y="cur_phi", hue=feat_filt_im[stain], 
                          xticklabels=np.around(theta_labs, 2), yticklabels=np.around(phi_labs, 2), 
                          vmin=vmin, vmax=vmax, ax=ax, cbar=False, cmap=sns.mpl_palette(col_map, 256), kind="kde")
        ax.set_ylabel("phi")
        ax.set_xlabel("theta")
        f1.colorbar(ax_, ax=ax, label=cbar_lab)
    
    f1.savefig(path_out_im+im+"_scatter_kde.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print("make plots --- %s seconds ---\n" % (time.time() - start_time_2))



