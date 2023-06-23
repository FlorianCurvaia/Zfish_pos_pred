#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:50:50 2023

@author: floriancurvaia
"""
import numpy as np

import pandas as pd

import math

import plotly.express as px

import matplotlib.pyplot as plt

from pathlib import Path


path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/New_theta/"
path_in_df="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/"

images=["B02_px-0280_py+1419", "B03_px-0545_py-1946", "C04_px-0816_py-1668", "B05_px+1522_py-1087", "C05_px+0198_py+1683", 
        "D05_px-0738_py-1248", "B06_px-0493_py+0146", "D06_px-1055_py-0118", "B07_px+1257_py-0474", "B08_px+1076_py-0189"]

images_2=['D06_px-0377_py-1358', 'D06_px+1741_py-0131', 'D06_px+0217_py+1735', 'D06_px-1055_py-0118', 
        'B07_px-0202_py+0631', 'B07_px+1257_py-0474', 'B07_px+0030_py-1959', 'B07_px-2056_py-0041', 
        'B07_px+0282_py+1729', 'C07_px+1425_py+0902', 'C07_px+1929_py-0176', 'C07_px-0784_py-0616', 
        'C07_px+0243_py-1998', 'C07_px+0301_py+1212', 'D07_px-0500_py+0670', 'D07_px+0999_py-1281']

images=list(set(images).union(set(images_2)))

time_emb={"B02":3.3, "C02":3.3, "D02":3.3, "B03":3.7, "C03":3.7, "D03":3.7, "B04":4.3, "C04":4.3, "D04":4.3,
          "B05":4.7, "C05":5.3, "D05":5.7, "B06":5.7, "C06":5.7, "D06":6.0, "B07":6.0, "C07":6.0, "D07":6.0,
          "B08":7.0, "C08":7.0, "D08":7.0}
images.remove("C07_px+0243_py-1998")
images.remove('B07_px-0202_py+0631')
to_remove=[]
colnames=['Embryo', 'Cell ID1', 'Cell ID2', 'Cell ID3', 'Cell ID4', 'Cell ID5', 'Cell ID6']
dorsal_cells=pd.read_csv("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/CellIDS.csv", sep=";", index_col=False, usecols=colnames)
curated_ids_emb={}
images=dorsal_cells.Embryo.tolist()
for im in images:
    dors_ids=dorsal_cells.loc[dorsal_cells.Embryo==im, dorsal_cells.columns != 'Embryo'].values.tolist()[0]
    if dorsal_cells.loc[dorsal_cells.Embryo==im].isnull().values.any():
        to_remove.append(im)
    else:
        curated_ids_emb[im]=dors_ids
images=list(set(images) - set(to_remove))

images=list(Path(path_in_df).glob("*.csv"))

t_max=[]
hpf=[]

for idx in range(len(images)):
    im=images[idx]
    im_name=im.name.split("_w")[0]
    well=im.name.split("_")[0]
    #fn_dist=path_in_df+im+"_w_dist_sph_simp.csv" #B08_px+1076_py-0189_w_dist_sph_simp.csv"
    feat_filt=pd.read_csv(im, sep=",", index_col=False)
    feat_filt["Theta"]=feat_filt.theta/np.max(feat_filt.theta)
    #feat_filt["Phi"]=feat_filt.cur_phi.abs()
    t_max.append(np.max(feat_filt.theta))
    hpf.append(time_emb[well])

    fig1 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="theta", opacity=1, color_continuous_scale="turbo", range_color=[0, math.pi])
    #fig1.update_traces(marker=dict(line=dict(width=10,color='black')))
    
    fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    #fig1.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
    fig1.write_html(path_out_im+im_name+"_theta.html")
    
    fig2 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="Theta", opacity=1, color_continuous_scale="turbo", range_color=[0, 1])
    #fig1.update_traces(marker=dict(line=dict(width=10,color='black')))
    
    fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    #fig1.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
    fig2.write_html(path_out_im+im_name+"_rel_theta.html")
    
    #fig3 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="Phi", opacity=1, color_continuous_scale="turbo", range_color=[0, math.pi])
    #fig3.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    #fig1.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
    #fig3.write_html(path_out_im+im+"_cur_phi_abs.html")

fig, ax= plt.subplots()
scatter=ax.scatter(hpf, t_max, c=hpf, cmap="viridis")
ax.set_ylabel("theta max")
ax.set_xlabel("hpf")
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower right", title="hpf")
ax.add_artist(legend1)
#ax.legend()
fig.savefig(path_out_im+"theta_max_per_emb.png", bbox_inches='tight', dpi=300) 
