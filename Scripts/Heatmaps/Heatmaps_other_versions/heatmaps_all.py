#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:40:37 2023

@author: floriancurvaia
"""

import numpy as np

import time

import pandas as pd

import math

import matplotlib.pyplot as plt

import argparse

from matplotlib.figure import figaspect


#B02_px-0280_py+1419, B03_px-0545_py-1946, B04_px+0114_py-1436, B05_px+1522_py-1087, C05_px+0198_py+1683, 
#D05_px-0738_py-1248, B06_px-0493_py+0146, D06_px-1055_py-0118, B07_px+1257_py-0474, B08_px+1076_py-0189
#C04_px-0816_py-1668 INSTEAD OF B04
start_time_0=time.time()
CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--image",  # name on the CLI - drop the `--` for positional/required parameters
  nargs=1,  # 0 or more values expected => creates a list
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
path_in_dist="/data/homes/fcurvaia/distances/"

path_out_im="/data/homes/fcurvaia/Images/Heatmaps/Sing_cells/"
im=args.image[0]
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
    
print(stains)

n_bins=30
r_sphere=np.load("/data/homes/fcurvaia/Spheres_fit/out_sphere_radius_"+im+".npy")
fn1=path_in_dist+im+"_w_dist_sph_simp.csv"
#adj_mat=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/"+im+"_adj_mat.npy")


feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "x_corr", "y_corr", "z_corr", "structure", "r_neigh", "r_neigh_mean", "NTouchingNeighbors","PhysicalSize", "theta","phi", "phi_bin", "theta_bin", "dist_out", "phi_bin_new", "phi_bin_cur", "new_phi", "cur_phi"]+stains)
#feat_filt=feat_filt[(feat_filt.PhysicalSize<12500)]

print("load files --- %s seconds ---\n" % (time.time() - start_time_0))


start_time_1=time.time()
for stain in stains:
    feat_filt=feat_filt.drop(feat_filt[feat_filt[stain]==0].index)

phi_bins=np.linspace(-math.pi, math.pi, n_bins)

theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)

## Phi-Theta heatmaps for 3 distance classes
#pSmad2/3_nc_ratio
dist_bins_ang=np.linspace(0, np.max(feat_filt.dist_out)+0.000000001, 4)

labels_d_b_a = range(1, 4)

t_max=np.max(feat_filt.theta_bin)

feat_filt['dist_bin_ang'] = pd.to_numeric(pd.cut(x = feat_filt['dist_out'], bins = dist_bins_ang, labels = labels_d_b_a, include_lowest = True, right=False))

n_bins_d=10

dist_bins=np.linspace(0, np.max(feat_filt.dist_out)+0.000000001, n_bins_d)

labels_d = range(1, n_bins_d)

feat_filt['dist_bin'] = pd.to_numeric(pd.cut(x = feat_filt['dist_out'], bins = dist_bins, labels = labels_d, include_lowest = True, right=False))

t_max=np.max(feat_filt.theta_bin)

theta_bins=theta_bins[:t_max]
theta_labs=360*theta_bins/(2*math.pi)
theta_labs=theta_labs#[:t_max]
phi_labs=360*phi_bins/(2*math.pi)
phi_labs=phi_labs


stains=stains+["r_neigh", "r_neigh_mean"]

l=len(dyes)
#w, h=figaspect(900/1440)
f, axs = plt.subplots(l,3, gridspec_kw={'width_ratios': [1, 1, 1.25]}, figsize=(30, 30))#
fon_size=15
space_size=38
plt.rcParams.update({'font.size': fon_size}) 
#sns.set_style("white",  {'figure.facecolor': 'white'})
plt.subplots_adjust(hspace=0.25)
f.set_dpi(300)
for s, ax in zip(range(l), axs):
    
    stain=dyes[s]
    #Phi Theta heatmaps

    stain_bins=np.linspace(0, np.max(feat_filt[stain]), 20)
    c=stain.split("_")
    stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
    vmin=feat_filt[stain].quantile(0.05)
    vmax=feat_filt[stain].quantile(0.95)
    
    
    ax[0].scatter(feat_filt.theta, feat_filt.cur_phi, c=feat_filt[stain], marker="o", s=16, vmin=vmin, vmax=vmax, cmap="jet", alpha=0.5)
    ax[0].set_xlabel("AP"+space_size*" "+" "+ space_size*" "+"Margin\n"+"Theta", fontsize=fon_size)
    ax[0].set_ylabel("Phi\n"+"V"+(space_size+5)*" "+"D"+(space_size+5)*" "+"V", fontsize=fon_size)
    ax[0].set_xticks(theta_bins, np.around(theta_labs, 2), rotation=90, fontsize=fon_size)
    ax[0].set_yticks(phi_bins, np.around(phi_labs, 2), fontsize=fon_size)
    #ax1.xticks(theta_bins, np.around(theta_labs, 2))
    #ax1.yticks(phi_bins, np.around(phi_labs, 2))
    #ax1.set_title("EVL")
    

    ax[1].scatter(feat_filt.cur_phi, feat_filt.dist_out, c=feat_filt[stain], marker="o", s=16, vmin=vmin, vmax=vmax, cmap="jet", alpha=0.5)
    ax[1].set_xlabel("V"+(space_size+2)*" "+"D"+(space_size+2)*" "+"V\n"+"Phi", fontsize=fon_size)
    ax[1].set_ylabel("Dist", fontsize=fon_size)
    ax[1].set_xticks(phi_bins, np.around(phi_labs, 2), rotation=90, fontsize=fon_size)
    ax[1].set_yticks(dist_bins, np.around(dist_bins, 2), fontsize=fon_size)
    #ax2.xticks(phi_bins, np.around(phi_labs, 2))
    #ax2.yticks(dist_bins, np.around(dist_bins, 2))
    #ax2.set_title("Middle")
    

    ax_=ax[2].scatter(feat_filt.theta, feat_filt.dist_out, c=feat_filt[stain], marker="o", s=16, vmin=vmin, vmax=vmax, cmap="jet", alpha=0.5)
    f.colorbar(ax_, ax=ax[2], label=stain+" intensity")
    ax[2].set_xlabel("AP"+space_size*" "+" "+ space_size*" "+"Margin\n"+"Theta", fontsize=fon_size)
    ax[2].set_ylabel("Dist", fontsize=fon_size)
    ax[2].set_xticks(theta_bins, np.around(theta_labs, 2), rotation=90, fontsize=fon_size)
    ax[2].set_yticks(dist_bins, np.around(dist_bins, 2), fontsize=fon_size)
    #ax3.xticks(theta_bins, np.around(theta_labs, 2))
    #ax3.yticks(dist_bins, np.around(dist_bins, 2))
    #ax3.set_title("YSL")
    
    
    
f.savefig(path_out_im+im+"_all_sing_cell.png", bbox_inches='tight', dpi=300)
plt.close()
    
    
    
    
    
    
    