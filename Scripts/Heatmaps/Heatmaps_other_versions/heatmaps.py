#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 09:10:51 2023

@author: floriancurvaia
"""

import numpy as np

import time

import pandas as pd

import sys

import math

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

start_time_0=time.time()


im="C05_px+0198_py+1683" #B04_px+0114_py-1436, B08_px+1076_py-0189, B07_px+1257_py-0474, B03_px-0545_py-1946, B02_px-0280_py+1419, "D06_px-1055_py-0118"  D06_px-1055_py-0118
n_bins=30
r_sphere=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/out_sphere_radius_"+im+".npy")
fn1="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv"
#adj_mat=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/"+im+"_adj_mat.npy")


feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "x_corr", "y_corr", "z_corr", "structure", "pSmad1/5_4_Mean", "pSmad1/5_nc_ratio", "betaCatenin_nc_ratio", "betaCatenin_nuc", "theta","phi", "phi_bin", "theta_bin", "dist_out", "MapK_nc_ratio", "MapK_2_Mean", "r_neigh", "NTouchingNeighbors"])
feat_filt=feat_filt.drop(feat_filt[feat_filt.MapK_nc_ratio==0].index)
print("load files --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_2=time.time()

phi_bins=np.linspace(-math.pi, math.pi, n_bins)

theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)

## Phi-Theta heatmaps for 3 distance classes
#pSmad2/3_nc_ratio
dist_bins_ang=np.linspace(0, np.max(feat_filt.dist_out)+0.000000001, 4)

labels_d_b_a = range(1, 4)

t_max=np.max(feat_filt.theta_bin)

feat_filt['dist_bin_ang'] = pd.to_numeric(pd.cut(x = feat_filt['dist_out'], bins = dist_bins_ang, labels = labels_d_b_a, include_lowest = True, right=False))

means_pSmad_15_p_t=feat_filt.groupby(['dist_bin_ang', 'phi_bin', 'theta_bin'])["MapK_nc_ratio"].mean()

to_heatmap_d_p_t=[np.full([n_bins, np.max(feat_filt.theta_bin)], np.nan), np.full([n_bins, np.max(feat_filt.theta_bin)], np.nan), np.full([n_bins, np.max(feat_filt.theta_bin)], np.nan)]

for idx in means_pSmad_15_p_t.index:
    idx_dist=idx[0]
    idx_phi=idx[1]
    idx_theta=idx[2]
    to_heatmap_d_p_t[idx_dist-1][idx_phi-1, idx_theta-1]=means_pSmad_15_p_t.loc[(idx_dist, idx_phi,idx_theta)]


means_pSmad_15_p_t_2=feat_filt.groupby(['phi_bin', 'theta_bin'])["MapK_nc_ratio"].mean()

to_heatmap_p_t=np.full([n_bins, np.max(feat_filt.theta_bin)], np.nan)

for idx in means_pSmad_15_p_t_2.index:
    idx_phi=idx[0]
    idx_theta=idx[1]
    to_heatmap_p_t[idx_phi-1, idx_theta-1]=means_pSmad_15_p_t_2.loc[(idx_phi,idx_theta)]


#Distances heatmaps

n_bins_d=10

dist_bins=np.linspace(0, np.max(feat_filt.dist_out)+0.000000001, n_bins_d)

labels_d = range(1, n_bins_d)

feat_filt['dist_bin'] = pd.to_numeric(pd.cut(x = feat_filt['dist_out'], bins = dist_bins, labels = labels_d, include_lowest = True, right=False))

#Distance-phi heatmap
means_pSmad_15_d_p=feat_filt.groupby(['dist_bin', 'phi_bin'])["MapK_nc_ratio"].mean()

to_heatmap_d_p=np.full([n_bins_d, n_bins], np.nan)

for idx in means_pSmad_15_d_p.index:
    idx_dist=idx[0]
    idx_phi=idx[1]
    to_heatmap_d_p[idx_dist-1, idx_phi-1]=means_pSmad_15_d_p.loc[(idx_dist, idx_phi)]


#Distance-theta heatmap
means_pSmad_15_d_t=feat_filt.groupby(['dist_bin', 'theta_bin'])["MapK_nc_ratio"].mean()

to_heatmap_d_t=np.full([n_bins_d, np.max(feat_filt.theta_bin)], np.nan)

for idx in means_pSmad_15_d_t.index:
    idx_dist=idx[0]
    idx_theta=idx[1]
    to_heatmap_d_t[idx_dist-1, idx_theta-1]=means_pSmad_15_d_t.loc[(idx_dist, idx_theta)]

min_sig=np.min(means_pSmad_15_p_t)#np.min(feat_filt["pSmad2/3_2_Mean"])
max_sig=np.max(means_pSmad_15_p_t)#np.max(feat_filt["pSmad2/3_2_Mean"])

bins_pop=feat_filt.groupby(['dist_bin', "theta_bin",'phi_bin']).size()

def volume_sections(distance_bins, r, n_bins_ang):
    vols_r_sect=dict()
    tot_ang_bins=n_bins_ang**2
    for i in range(len(distance_bins)-1):
        vol1=4/3*math.pi*(r-distance_bins[i])**3
        vol2=4/3*math.pi*(r-distance_bins[i+1])**3
        sect_vol=(vol1-vol2)/tot_ang_bins
        vols_r_sect[i+1]=sect_vol
    return vols_r_sect
vol_by_sector=volume_sections(dist_bins, r_sphere, n_bins)
feat_filt["sect_pop"]=pd.MultiIndex.from_frame(feat_filt[["dist_bin", "theta_bin", "phi_bin"]]).map(bins_pop)
feat_filt["sect_vol"]=feat_filt["dist_bin"].map(vol_by_sector)
feat_filt["sect_dens"]=feat_filt.sect_pop/feat_filt.sect_vol
print("Generate heatmap matrix --- %s seconds ---\n" % (time.time() - start_time_2))

path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/D_in_microns/"

theta_labs=360*theta_bins/(2*math.pi)
theta_labs=theta_labs[:t_max]
phi_labs=360*phi_bins/(2*math.pi)
phi_labs=phi_labs[:-1]

f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)
g1 = sns.heatmap(to_heatmap_d_p_t[0], linewidth=0.5, xticklabels=np.around(theta_labs, 2), yticklabels=np.around(phi_labs, 2), vmin=min_sig, vmax=max_sig, ax=ax1, cbar=False, cmap="viridis")
g1.set_ylabel("phi")
g1.set_xlabel("theta")
g1.set_title("EVL")
g2 = sns.heatmap(to_heatmap_d_p_t[1], linewidth=0.5, xticklabels=np.around(theta_labs, 2), yticklabels=np.around(phi_labs, 2), vmin=min_sig, vmax=max_sig, ax=ax2, cbar=False, cmap="viridis") 
g2.set_ylabel("phi")
g2.set_xlabel("theta")
g2.set_title("Middle")
g3 = sns.heatmap(to_heatmap_d_p_t[2], linewidth=0.5, xticklabels=np.around(theta_labs, 2), yticklabels=np.around(phi_labs, 2), vmin=min_sig, vmax=max_sig, ax=ax3,  cmap="viridis") 
g3.collections[0].colorbar.set_label("pSmad1/5 nc ratio Mean intensity")
g3.set_ylabel("phi")
g3.set_xlabel("theta")
g3.set_title("YSL")
plt.savefig(path_out_im+im+"_"+"stain_clean"+"_dist_phi_theta.png")



"""
fig0, ax0 = plt.subplots(2, 2)
plt.clf()
ax0[0,0] = sns.heatmap(to_heatmap_d_p_t[0], linewidth=0.5, xticklabels=np.around(theta_bins, 2), yticklabels=np.around(theta_bins, 2), vmin=min_sig, vmax=max_sig) #, cbar_kws={'label': 'pSmad2/3 Mean intensity'}
ax0[0,0].collections[0].colorbar.set_label("pSmad1/5 nc ratio Mean intensity_EVL")
ax0[0,0].set(xlabel="theta", ylabel="phi")

ax0[0,1] = sns.heatmap(to_heatmap_d_p_t[1], linewidth=0.5, xticklabels=np.around(theta_bins, 2), yticklabels=np.around(theta_bins, 2), vmin=min_sig, vmax=max_sig) #, cbar_kws={'label': 'pSmad2/3 Mean intensity'}
ax0[0,1].collections[0].colorbar.set_label("pSmad1/5 nc ratio Mean intensity_Mid")
ax0[0,1].set(xlabel="theta", ylabel="phi")

ax0[1,0] = sns.heatmap(to_heatmap_d_p_t[2], linewidth=0.5, xticklabels=np.around(theta_bins, 2), yticklabels=np.around(theta_bins, 2), vmin=min_sig, vmax=max_sig) #, cbar_kws={'label': 'pSmad2/3 Mean intensity'}
ax0[1,0].collections[0].colorbar.set_label("pSmad1/5 nc ratio Mean intensity_YSL")
ax0[1,0].set(xlabel="theta", ylabel="phi")


#for hm in range(len(to_heatmap)):
fig1=plt.figure(1)
plt.clf()
ax = sns.heatmap(to_heatmap_d_p_t[0], linewidth=0.5, xticklabels=np.around(theta_bins[:t_max], 2), yticklabels=np.around(phi_bins, 2), vmin=min_sig, vmax=max_sig) #, cbar_kws={'label': 'pSmad2/3 Mean intensity'}
ax.collections[0].colorbar.set_label("pSmad1/5 nc ratio Mean intensity_EVL")
plt.xlabel("theta")
plt.ylabel("phi")

fig2=plt.figure(2)
plt.clf()
ax = sns.heatmap(to_heatmap_d_p_t[1], linewidth=0.5, xticklabels=np.around(theta_bins[:t_max], 2), yticklabels=np.around(phi_bins, 2), vmin=min_sig, vmax=max_sig) #, cbar_kws={'label': 'pSmad2/3 Mean intensity'}
ax.collections[0].colorbar.set_label("pSmad1/5 nc ratio Mean intensity_Mid")
plt.xlabel("theta")
plt.ylabel("phi")

fig3=plt.figure(3)
plt.clf()
ax = sns.heatmap(to_heatmap_d_p_t[2], linewidth=0.5, xticklabels=np.around(theta_bins[:t_max], 2), yticklabels=np.around(phi_bins, 2), vmin=min_sig, vmax=max_sig) #, cbar_kws={'label': 'pSmad2/3 Mean intensity'}
ax.collections[0].colorbar.set_label("pSmad1/5 nc ratio Mean intensity_YSL")
plt.xlabel("theta")
plt.ylabel("phi")
"""

fig4=plt.figure(4)
plt.clf()
ax = sns.heatmap(to_heatmap_d_p, linewidth=0.5, xticklabels=np.around(phi_labs , 2), yticklabels=np.around(dist_bins[:-1], 2),) #, cbar_kws={'label': 'pSmad2/3 Mean intensity'}
ax.collections[0].colorbar.set_label("pSmad1/5 nc ratio Mean intensity")
plt.xlabel("phi")
plt.ylabel("dist")

fig5=plt.figure(5)
plt.clf()
ax = sns.heatmap(to_heatmap_d_t, linewidth=0.5, xticklabels=np.around(theta_labs, 2), yticklabels=np.around(dist_bins[:-1], 2)) #, cbar_kws={'label': 'pSmad2/3 Mean intensity'}
ax.collections[0].colorbar.set_label("pSmad1/5 nc ratio Mean intensity")
plt.xlabel("theta")
plt.ylabel("dist")

means_ov_phi=feat_filt.groupby(['phi_bin'])["MapK_nc_ratio"].mean()

fig6=plt.figure(6)
plt.clf()
plt.plot(phi_labs, means_ov_phi)
plt.xlabel("phi")
plt.ylabel("MapK")

means_ov_theta=feat_filt.groupby(['theta_bin'])["MapK_nc_ratio"].mean()

fig7=plt.figure(7)
plt.clf()
plt.plot(theta_labs, means_ov_theta)
plt.xlabel("theta")
plt.ylabel("MapK")


fig8 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="sect_dens", opacity=1, color_continuous_scale="turbo")
fig8.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig8.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig8.write_html(path_out_im+"sect_dens_"+im+".html")

fig9=plt.figure(9)
plt.clf()
ax = sns.heatmap(to_heatmap_p_t, linewidth=0.5, xticklabels=np.around(theta_labs, 2), yticklabels=np.around(phi_labs, 2), cmap=sns.mpl_palette("turbo", 256)) #, cbar_kws={'label': 'pSmad2/3 Mean intensity'}
ax.collections[0].colorbar.set_label("pSmad1/5 nc ratio Mean intensity")
plt.xlabel("theta")
plt.ylabel("phi")

plt.show()
