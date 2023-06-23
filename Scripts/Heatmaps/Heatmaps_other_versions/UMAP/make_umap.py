#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:19:36 2023

@author: floriancurvaia
"""

import numpy as np

import time

import pandas as pd

import math

import umap

import matplotlib.pyplot as plt

import argparse

import plotly.express as px

from sklearn.manifold import TSNE

import seaborn as sns

from scipy.stats import zscore

import pacmap

from sklearn.decomposition import PCA

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

fn1=path_in_dist+im+"_w_dist_sph_simp.csv"
#adj_mat=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/"+im+"_adj_mat.npy")

#fn1="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv"
feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "x_corr", "y_corr", "z_corr", "structure", "r_neigh", "r_neigh_mean", "NTouchingNeighbors","PhysicalSize", "theta","phi", "phi_bin", "theta_bin", "dist_out", "phi_bin_new", "phi_bin_cur", "new_phi", "cur_phi"]+stains)


for col in dyes:
    feat_filt[col]=zscore(feat_filt[col], nan_policy="omit")

print("load files --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()

feat_um=feat_filt[dyes]

#feat_um=zscore(feat_um, nan_policy="omit")

umap_2d = umap.UMAP(n_components=2, random_state=0, n_neighbors=12)

proj_2d = umap_2d.fit_transform(feat_um)
"""

tsne_2d = TSNE(n_components=2, n_jobs=5, init="pca")

tsne_proj=tsne_2d.fit_transform(feat_um)
"""
pacmap_2d=pacmap.PaCMAP(random_state=42)

pacmap_proj=pacmap_2d.fit_transform(feat_um.to_numpy(), init="pca")

"""
pca_2d = PCA(n_components=2)
pca_proj = pca_2d.fit_transform(feat_um)


print("PCA explained variance by 1st component : {}".format(pca_2d.explained_variance_ratio_[0]))
print("PCA explained variance by 2nd component : {}".format(pca_2d.explained_variance_ratio_[1]))
"""
print("make umap --- %s seconds ---\n" % (time.time() - start_time_1))

start_time_2=time.time()
path_out_im="/data/homes/fcurvaia/Images/UMAP/All/"

prop=["cur_phi", "theta", "dist_out", "r_neigh"]

dyes+=prop

plt.style.use('dark_background')
l=len(dyes)
#w, h=figaspect(900/1440)
f, axs = plt.subplots(2,4, gridspec_kw={'width_ratios': [1, 1, 1, 1]}, figsize=(60, 30))#
fon_size=15
space_size=38
plt.rcParams.update({'font.size': fon_size}) 
#sns.set_style("white",  {'figure.facecolor': 'white'})
plt.subplots_adjust(hspace=0.125)
plt.subplots_adjust(wspace=0.125)
f.set_dpi(300)
for s, ax in zip(range(l), axs.flatten()):
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
        vmin=feat_filt[stain].min()
        vmax=feat_filt[stain].max()
        cbar_lab=stain
    elif stain in prop:
        vmin=feat_filt[stain].quantile(0.01)
        vmax=feat_filt[stain].quantile(0.99)
        cbar_lab=stain
    else:
        vmin=feat_filt[stain].quantile(0.01)
        vmax=feat_filt[stain].quantile(0.99)
        cbar_lab=stain+" intensity"
    
    ax_=ax.scatter(
         x=proj_2d[:,0], y=proj_2d[:,1], #data=proj_2d,
        c=feat_filt[stain], vmin=vmin, vmax=vmax, marker="o", s=16, cmap=col_map, alpha=0.5)
    ax.set_title(stain_clean)
    f.colorbar(ax_, ax=ax, label=cbar_lab)

f.savefig(path_out_im+im+"_umap_all.png", bbox_inches='tight', dpi=300)
plt.close()

f1, axs1 = plt.subplots(2,4, gridspec_kw={'width_ratios': [1, 1, 1, 1]}, figsize=(60, 30))#
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
        vmin=feat_filt[stain].min()
        vmax=feat_filt[stain].max()
        cbar_lab=stain
    elif stain in prop:
        vmin=feat_filt[stain].quantile(0.01)
        vmax=feat_filt[stain].quantile(0.99)
        cbar_lab=stain
    else:
        vmin=feat_filt[stain].quantile(0.01)
        vmax=feat_filt[stain].quantile(0.99)
        cbar_lab=stain+" intensity"
    
    ax_=ax.scatter(
        x=pacmap_proj[:,0], y=pacmap_proj[:,1], #data=pacmap_proj
        c=feat_filt[stain], vmin=vmin, vmax=vmax, marker="o", s=16, cmap=col_map, alpha=0.5)
    ax.set_title(stain_clean)
    f1.colorbar(ax_, ax=ax, label=cbar_lab)

f1.savefig(path_out_im+im+"_pacmap_all.png", bbox_inches='tight', dpi=300)
plt.close()

print("make plots --- %s seconds ---\n" % (time.time() - start_time_2))
"""
fig_2d_1 = px.scatter(
    proj_2d, x=0, y=1,
    color=feat_filt.cur_phi, labels={'color': 'Phi'})
fig_2d_1.write_html(path_out_im+"umap_new_phi_"+im+".html")

fig_2d_2 = px.scatter(
    proj_2d, x=0, y=1,
    color=feat_filt.theta, labels={'color': 'Theta'} )
fig_2d_2.write_html(path_out_im+"umap_theta_"+im+".html")

fig_2d_3 = px.scatter(
    proj_2d, x=0, y=1,
    color=feat_filt.dist_out, labels={'color': 'Dist_out'} )
fig_2d_3.write_html(path_out_im+"umap_dist_out_"+im+".html")
"""

"""
for stain in dyes:
    vmin=feat_filt[stain].quantile(0.05)
    vmax=feat_filt[stain].quantile(0.95)
    stain_clean="_".join(["".join(stain.split("/"))])
    fig_2d_4 = px.scatter(
        proj_2d, x=0, y=1,
        color=feat_filt[stain], labels={'color': stain}, range_color=[vmin, vmax] )
    fig_2d_4.write_html(path_out_im+"umap_"+stain_clean+"_"+im+".html")
    fig_2d_5 = px.scatter(
        pacmap_proj, x=0, y=1,
        color=feat_filt[stain], labels={'color': stain}, range_color=[vmin, vmax] )
    fig_2d_5.write_html(path_out_im+"pacmap_"+stain_clean+"_"+im+".html")
    fig_2d_6 = px.scatter(
        pca_proj, x=0, y=1,
        color=feat_filt[stain], labels={'color': stain}, range_color=[vmin, vmax] )
    fig_2d_6.write_html(path_out_im+"pca_"+stain_clean+"_"+im+".html")
"""


"""
vmin=feat_filt["pSmad1/5_nc_ratio"].quantile(0.05)
vmax=feat_filt["pSmad1/5_nc_ratio"].quantile(0.95)

fig_2d_4 = px.scatter(
    proj_2d, x=0, y=1,
    color=feat_filt["pSmad1/5_nc_ratio"], labels={'color': 'pSmad1/5_nc_ratio'}, range_color=[vmin, vmax] )
fig_2d_4.write_html(path_out_im+"umap_pSmad15_nc_ratio_"+im+".html")


vmin=feat_filt["betaCatenin_nuc"].quantile(0.05)
vmax=feat_filt["betaCatenin_nuc"].quantile(0.95)

fig_2d_5 = px.scatter(
    proj_2d, x=0, y=1,
    color=feat_filt["betaCatenin_nuc"], labels={'color': 'betaCatenin_nuc'}, range_color=[vmin, vmax] )
fig_2d_5.write_html(path_out_im+"umap_betaCatenin_nuc_"+im+".html")


vmin=feat_filt["MapK_nc_ratio"].quantile(0.05)
vmax=feat_filt["MapK_nc_ratio"].quantile(0.95)

fig_2d_6 = px.scatter(
    proj_2d, x=0, y=1,
    color=feat_filt["MapK_nc_ratio"], labels={'color': 'MapK_nc_ratio'}, range_color=[vmin, vmax] )
fig_2d_6.write_html(path_out_im+"umap_MapK_nc_ratio_"+im+".html")

feat_um["comp-1"]=tsne_proj[:,0]
feat_um["comp-2"]=tsne_proj[:,1]
"""
"""
fig1=plt.figure()
plt.clf()
fig_tsne_1 = sns.scatterplot(
    data=feat_um, x="comp-1", y="comp-2",
    hue=feat_filt.cur_phi,cmap=sns.mpl_palette("turbo", 256))
#fig_tsne_1.write_html(path_out_im+"tsne_new_phi_"+im+".html")
fig1.savefig(path_out_im+"tsne_new_phi_"+im+".png", bbox_inches='tight', dpi=300)
plt.close(fig1)

fig2=plt.figure()
plt.clf()
fig_tsne_2 = sns.scatterplot(
    data=feat_um, x="comp-1", y="comp-2",
    hue=feat_filt.theta, cmap=sns.mpl_palette("turbo", 256) )
#fig_tsne_2.write_html(path_out_im+"tsne_theta_"+im+".html")
fig2.savefig(path_out_im+"tsne_theta_"+im+".png", bbox_inches='tight', dpi=300)
plt.close(fig2)

fig3=plt.figure()
plt.clf()
fig_tsne_3 = sns.scatterplot(
    data=feat_um, x="comp-1", y="comp-2",
    hue=feat_filt.dist_out, cmap=sns.mpl_palette("turbo", 256) )
#fig_tsne_3.write_html(path_out_im+"tsne_dist_out_"+im+".html")
fig3.savefig(path_out_im+"tsne_dist_out_"+im+".png", bbox_inches='tight', dpi=300)
plt.close(fig3)
"""




