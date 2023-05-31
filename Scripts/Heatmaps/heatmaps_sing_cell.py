#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:42:22 2023

@author: floriancurvaia
"""



import numpy as np

import time

import pandas as pd

import math

import seaborn as sns

import matplotlib.pyplot as plt

import argparse

from pathlib import Path

#from multiprocessing import Pool

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

path_out_im="/data/homes/fcurvaia/Images/Heatmaps/"
im=args.image[0]
print(args.colnames)
stains=[]
for col in args.colnames:
    stain=col.split("_")[0]
    ratio=stain+"_nc_ratio"
    nuc=stain+"_nuc"
    cyto=stain+"_cyto"
    diff=stain+"_nc_diff"
    diff_ratio=stain+"_nc_diff_ratio"
    Path(path_out_im+"").mkdir(parents=True, exist_ok=True)
    stains.append(col)
    stains.append(ratio)
    stains.append(nuc)
    stains.append(cyto)
    stains.append(diff)
    stains.append(diff_ratio)
print(stains)

n_bins=30
r_sphere=np.load("/data/homes/fcurvaia/Spheres_fit/out_sphere_radius_"+im+".npy")
fn1=path_in_dist+im+"_w_dist_sph_simp.csv"
#adj_mat=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/"+im+"_adj_mat.npy")


feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "x_corr", "y_corr", "z_corr", "structure", "r_neigh", "r_neigh_mean", "NTouchingNeighbors","PhysicalSize", "theta","phi", "phi_bin", "theta_bin", "dist_out", "phi_bin_new", "phi_bin_cur", "new_phi"]+stains)
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


theta_labs=360*theta_bins/(2*math.pi)
theta_labs=theta_labs[:t_max]
phi_labs=360*phi_bins/(2*math.pi)
phi_labs=phi_labs[:-1]

stains=stains+["r_neigh", "r_neigh_mean"]
for stain in stains:
    c=stain.split("_")
    stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
    #Phi Theta heatmaps
    
    means_dist_phi_theta=feat_filt.groupby(['dist_bin_ang', 'phi_bin', 'theta_bin'])[stain].mean()
    
    heatmap_dist_phi_theta=[np.full([n_bins, np.max(feat_filt.theta_bin)], np.nan), np.full([n_bins, np.max(feat_filt.theta_bin)], np.nan), np.full([n_bins, np.max(feat_filt.theta_bin)], np.nan)]
    
    for idx in means_dist_phi_theta.index:
        idx_dist=idx[0]
        idx_phi=idx[1]
        idx_theta=idx[2]
        heatmap_dist_phi_theta[idx_dist-1][idx_phi-1, idx_theta-1]=means_dist_phi_theta.loc[(idx_dist, idx_phi,idx_theta)]
    
    min_sig=np.min(means_dist_phi_theta)
    max_sig=np.max(means_dist_phi_theta)
    
    
    f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True, gridspec_kw={'width_ratios': [1, 1, 1.25]})
    sns.set(font_scale=0.75)
    sns.set_style("white",  {'figure.facecolor': 'white'})
    plt.subplots_adjust(hspace=0.5)
    g1 = sns.heatmap(heatmap_dist_phi_theta[0], linewidth=0.5, xticklabels=np.around(theta_labs, 2), yticklabels=np.around(phi_labs, 2), vmin=min_sig, vmax=max_sig, ax=ax1, cbar=False, cmap=sns.mpl_palette("turbo", 256))
    g1.set_ylabel("phi")
    g1.set_xlabel("theta")
    g1.set_title("EVL")
    g2 = sns.heatmap(heatmap_dist_phi_theta[1], linewidth=0.5, xticklabels=np.around(theta_labs, 2), yticklabels=np.around(phi_labs, 2),  vmin=min_sig, vmax=max_sig, ax=ax2, cbar=False, cmap=sns.mpl_palette("turbo", 256)) #yticklabels=np.around(phi_labs, 2),
    #g2.set_ylabel("phi")
    g2.set_xlabel("theta")
    g2.set_title("Middle")
    g3 = sns.heatmap(heatmap_dist_phi_theta[2], linewidth=0.5, xticklabels=np.around(theta_labs, 2), yticklabels=np.around(phi_labs, 2), vmin=min_sig, vmax=max_sig, ax=ax3,  cmap=sns.mpl_palette("turbo", 256)) # yticklabels=np.around(phi_labs, 2),
    g3.collections[0].colorbar.set_label(stain+" intensity")
    #g3.set_ylabel("phi")
    g3.set_xlabel("theta")
    g3.set_title("YSL")
    f.savefig(path_out_im+im+"_"+stain_clean+"_dist_phi_theta.png", bbox_inches='tight', dpi=300)
    plt.close(f)
    
    
    
    means_phi_theta=feat_filt.groupby(['phi_bin_new', 'theta_bin'])[stain].mean()
    
    heatmap_phi_theta=np.full([n_bins, np.max(feat_filt.theta_bin)], np.nan)
    
    for idx in means_phi_theta.index:
        idx_phi=idx[0]
        idx_theta=idx[1]
        heatmap_phi_theta[idx_phi-1, idx_theta-1]=means_phi_theta.loc[(idx_phi,idx_theta)]
    
    
    
   
    fig0=plt.figure(0)
    plt.clf()
    vmin=feat_filt[stain].quantile(0.05)
    vmax=feat_filt[stain].quantile(0.95)
    ax=plt.scatter(feat_filt.theta, feat_filt.new_phi, c=feat_filt[stain], marker=".", vmin=vmin, vmax=vmax, cmap="turbo")
    plt.colorbar(ax)
    plt.xlabel("theta")
    plt.ylabel("phi")
    fig0.savefig(path_out_im+im+"_"+stain_clean+"_phi_new_theta_sing_cell.png", bbox_inches='tight', dpi=300)
    plt.close(fig0)
    """
    vmin=feat_filt[stain].quantile(0.05)
    vmax=feat_filt[stain].quantile(0.95)
    heatmap_phi_theta=feat_filt.pivot(index='new_phi', columns='theta', values=stain)
    ax = sns.heatmap(heatmap_phi_theta, linewidth=0.5, xticklabels=np.around(theta_labs, 2), yticklabels=np.around(phi_labs, 2),vmin=vmin, vmax=vmax, cmap=sns.mpl_palette("turbo", 256)) 
    ax.collections[0].colorbar.set_label(stain+" intensity")
    plt.xlabel("theta")
    plt.ylabel("phi")
    fig1.savefig(path_out_im+im+"_"+stain_clean+"_phi_new_theta_sing_cell.png", bbox_inches='tight', dpi=300)
    plt.close(fig1)
    """
    
    means_phi_theta=feat_filt.groupby(['phi_bin_cur', 'theta_bin'])[stain].mean()
    
    heatmap_phi_theta=np.full([n_bins, np.max(feat_filt.theta_bin)], np.nan)
    
    for idx in means_phi_theta.index:
        idx_phi=idx[0]
        idx_theta=idx[1]
        heatmap_phi_theta[idx_phi-1, idx_theta-1]=means_phi_theta.loc[(idx_phi,idx_theta)]
        
    fig1=plt.figure()
    plt.clf()
    vmin=feat_filt[stain].quantile(0.05)
    vmax=feat_filt[stain].quantile(0.95)
    ax = sns.heatmap(heatmap_phi_theta, linewidth=0.5, xticklabels=np.around(theta_labs, 2), yticklabels=np.around(phi_labs, 2),vmin=vmin, vmax=vmax, cmap=sns.mpl_palette("turbo", 256)) 
    ax.collections[0].colorbar.set_label(stain+" intensity")
    plt.xlabel("theta")
    plt.ylabel("phi")
    fig1.savefig(path_out_im+im+"_"+stain_clean+"_phi_cur_theta.png", bbox_inches='tight', dpi=300)
    plt.close(fig1)
    #Distance-phi heatmap
    means_dist_phi=feat_filt.groupby(['dist_bin', 'phi_bin'])[stain].mean()
    
    heatmap_dist_phi=np.full([n_bins_d, n_bins], np.nan)
    
    for idx in means_dist_phi.index:
        idx_dist=idx[0]
        idx_phi=idx[1]
        heatmap_dist_phi[idx_dist-1, idx_phi-1]=means_dist_phi.loc[(idx_dist, idx_phi)]
        
    fig2=plt.figure()
    plt.clf()
    ax = sns.heatmap(heatmap_dist_phi, linewidth=0.5, xticklabels=np.around(phi_labs , 2), yticklabels=np.around(dist_bins[:-1], 2), cmap=sns.mpl_palette("turbo", 256)) 
    ax.collections[0].colorbar.set_label(stain+" intensity")
    plt.xlabel("phi")
    plt.ylabel("dist")
    fig2.savefig(path_out_im+im+"_"+stain_clean+"_dist_phi.png", bbox_inches='tight', dpi=300)
    plt.close(fig2)
    
    #Distance-theta heatmap
    means_dist_theta=feat_filt.groupby(['dist_bin', 'theta_bin'])[stain].mean()
    
    heatmap_dist_theta=np.full([n_bins_d, np.max(feat_filt.theta_bin)], np.nan)
    
    for idx in means_dist_theta.index:
        idx_dist=idx[0]
        idx_theta=idx[1]
        heatmap_dist_theta[idx_dist-1, idx_theta-1]=means_dist_theta.loc[(idx_dist, idx_theta)]
    
    fig3=plt.figure()
    plt.clf()
    ax = sns.heatmap(heatmap_dist_theta, linewidth=0.5, xticklabels=np.around(theta_labs , 2), yticklabels=np.around(dist_bins[:-1], 2), cmap=sns.mpl_palette("turbo", 256)) 
    ax.collections[0].colorbar.set_label(stain+" intensity")
    plt.xlabel("theta")
    plt.ylabel("dist")
    fig3.savefig(path_out_im+im+"_"+stain_clean+"_dist_theta.png", bbox_inches='tight', dpi=300)
    plt.close(fig3)
    
    means_ov_phi=feat_filt.groupby(['phi_bin'])[stain].mean()

    fig4=plt.figure()
    plt.clf()
    plt.plot(phi_labs, means_ov_phi)
    plt.xlabel("phi")
    plt.ylabel(stain)
    fig4.savefig(path_out_im+im+"_"+stain_clean+"_means_ov_phi.png", bbox_inches='tight', dpi=300)
    plt.close(fig4)

    means_ov_theta=feat_filt.groupby(['theta_bin'])[stain].mean()

    fig5=plt.figure()
    plt.clf()
    plt.plot(theta_labs, means_ov_theta)
    plt.xlabel("theta")
    plt.ylabel(stain)
    fig5.savefig(path_out_im+im+"_"+stain_clean+"_means_ov_theta.png", bbox_inches='tight', dpi=300)
    plt.close(fig5)



print("Generate heatmaps --- %s seconds ---\n" % (time.time() - start_time_1))



