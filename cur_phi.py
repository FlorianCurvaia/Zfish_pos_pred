#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:12:33 2023

@author: floriancurvaia
"""



import numpy as np

import time

import pandas as pd

import math

from itertools import combinations

import plotly.express as px

from pathlib import Path

path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/DV_axis/Opt/"
path_out_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/"

path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/pre_aligned/Images/DV_axis/Opt/"
path_out_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/pre_aligned/distances/"

path_csv=Path(path_out_dist)

#images=["B02_px-0280_py+1419", "B03_px-0545_py-1946", "C04_px-0816_py-1668", "B05_px+1522_py-1087", "C05_px+0198_py+1683", 
#        "D05_px-0738_py-1248", "B06_px-0493_py+0146", "D06_px-1055_py-0118", "B07_px+1257_py-0474", "B08_px+1076_py-0189"]

#images=['D06_px-0377_py-1358', 'D06_px+1741_py-0131', 'D06_px+0217_py+1735', 'D06_px-1055_py-0118', 
        #'B07_px-0202_py+0631', 'B07_px+1257_py-0474', 'B07_px+0030_py-1959', 'B07_px-2056_py-0041', 
        #'B07_px+0282_py+1729', 'C07_px+1425_py+0902', 'C07_px+1929_py-0176', 'C07_px-0784_py-0616', 
        #'C07_px+0243_py-1998', 'C07_px+0301_py+1212', 'D07_px-0500_py+0670', 'D07_px+0999_py-1281']

#images=["C04_px-0816_py-1668"]

#images_1=["B02_px-0280_py+1419", "B03_px-0545_py-1946", "C04_px-0816_py-1668", "B05_px+1522_py-1087", "C05_px+0198_py+1683", 
#        "D05_px-0738_py-1248", "B06_px-0493_py+0146", "D06_px-1055_py-0118", "B07_px+1257_py-0474", "B08_px+1076_py-0189"]
#images+=images_1

#images.remove('B07_px-0202_py+0631')
#images.remove("C07_px+0243_py-1998")
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
"""
curated_ids_emb["B02"]=[1877, 1807, 1619, 2001, 1620, 1664] #B02
curated_ids_emb["B03"]=[4325, 4362, 4585, 4563, 4657, 4068] #B03
curated_ids_emb["C04"]=[4284, 3782, 4283, 4173, 4312, 3697] #C04
curated_ids_emb["B05"]=[5891, 6008, 5946, 5763, 6011, 5838] #B05
curated_ids_emb["C05"]=[5212, 5056, 5174, 4950, 5422, 4407] #C05
curated_ids_emb["D05"]=[7493, 7553, 7492, 7243, 7348, 7460] #D05
curated_ids_emb["B06"]=[4210, 4372, 3749, 3747, 3467, 4294] #B06
curated_ids_emb["D06"]=[9070, 9128, 7754, 9726, 8243, 8394] #D06
curated_ids_emb["B07"]=[6485, 6919, 7266, 6300, 6799, 6838] #B07
curated_ids_emb["B08"]=[5756, 6708, 6455, 6282, 6366, 5830] #B08
"""
#B02_px-0280_py+1419, B03_px-0545_py-1946, C04_px-0816_py-1668, B05_px+1522_py-1087, C05_px+0198_py+1683, 
#D05_px-0738_py-1248, B06_px-0493_py+0146, D06_px-1055_py-0118, B07_px+1257_py-0474, B08_px+1076_py-0189
#C04_px-0816_py-1668 INSTEAD OF B04_px+0114_py-1436

images=list(set(images) - set(to_remove))
n_bins=20
cur_max_dist={}

images=list(path_csv.glob("*.csv"))

#images=["C05_px+0198_py+1683"]

#for im_path in images:
#    im=im_path.name.split("_w")[0]
    
for idx in range(len(images)):
    im=images[idx].name.split("_w")[0]
    row_to_fill=[]
    fn_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/"+im+"_w_dist_sph_simp.csv" #B08_px+1076_py-0189_w_dist_sph_simp.csv"
    feat_filt=pd.read_csv(fn_dist, sep=",", index_col=False)
    #feat_filt=pd.read_csv(im_path, sep=",", index_col=False)
    
    
    start_time_1=time.time()
    well=im.split("_")[0]
    row_to_fill.append(well)
    curated_ids=curated_ids_emb[im]
    mean_phi_cur=feat_filt.loc[feat_filt.Label.isin(curated_ids)]["phi"].mean()
    diff=[]
    for i in combinations(feat_filt.loc[feat_filt.Label.isin(curated_ids)]["phi"].to_numpy(), 2):
        d=math.sqrt((i[0]-i[1])**2)
        d=360*d/(2*math.pi)
        diff.append(d)
    cur_max_dist[im]=max(diff)
   
    cur_phi_0=mean_phi_cur

    cur_phi_sign=np.sign(cur_phi_0)
    if cur_phi_sign==-1:
        feat_filt["cur_phi"]=np.abs(cur_phi_0)+feat_filt["phi"]
        feat_filt.loc[(feat_filt["cur_phi"]>math.pi), "cur_phi"]=feat_filt["cur_phi"]-2*math.pi
    elif cur_phi_sign==1:
        feat_filt["cur_phi"]=feat_filt["phi"]-cur_phi_0
        feat_filt.loc[(feat_filt["cur_phi"]<-1*math.pi), "cur_phi"]=feat_filt["cur_phi"]+2*math.pi
    else:
        feat_filt["cur_phi"]=feat_filt["phi"]

    feat_filt["rel_theta"]=feat_filt.theta/np.max(feat_filt.theta)
    phi_bins=np.linspace(-math.pi, math.pi, n_bins)
    labels = range(1, n_bins)

    feat_filt['phi_bin_cur'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
    
    feat_filt.to_csv(path_out_dist+im+"_w_dist_sph_simp.csv", sep=",", index=False)
    
    fig1 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="cur_phi", opacity=1, color_continuous_scale="turbo")
    #fig1.update_traces(marker=dict(line=dict(width=10,color='black')))
    fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0), coloraxis_colorbar_title_text = 'Phi')
    fig1.write_html(path_out_im+"cur_phi_"+im+".html")
    
    fig2 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="rel_theta", opacity=1, color_continuous_scale="turbo")
    #fig2.update_traces(marker=dict(line=dict(width=10,color='black')))
    fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0), coloraxis_colorbar_title_text = 'Theta')
    fig2.write_html(path_out_im+"rel_theta_"+im+".html")
    
    
    
    