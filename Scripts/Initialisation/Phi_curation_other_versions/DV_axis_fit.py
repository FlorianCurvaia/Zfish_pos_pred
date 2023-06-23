#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:44:47 2023

@author: floriancurvaia
"""


import numpy as np

import time

import pandas as pd

import math

import plotly.express as px

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
import scipy.stats as stats

margin=0.8
stain="betaCatenin_nuc" #"pSmad1/5_nc_ratio"
start_time_0=time.time()
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/DV_axis/Opt/"
path_out_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"
im="B07_px+1257_py-0474" 
#B02_px-0280_py+1419, B03_px-0545_py-1946, B04_px+0114_py-1436, B05_px+1522_py-1087, C05_px+0198_py+1683, 
#D05_px-0738_py-1248, B06_px-0493_py+0146, D06_px-1055_py-0118, B07_px+1257_py-0474, B08_px+1076_py-0189
#C04_px-0816_py-1668 INSTEAD OF B04
fn_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv" #B08_px+1076_py-0189_w_dist_sph_simp.csv"
feat_filt=pd.read_csv(fn_dist, sep=",", index_col=False)
n_bins=36
shifts=np.linspace(0, math.pi, 60)
print("load files --- %s seconds ---\n" % (time.time() - start_time_0))


start_time_1=time.time()
well=im.split("_")[0]
curated_ids_emb={}
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
curated_ids=curated_ids_emb[well]
mean_phi_cur=feat_filt.loc[feat_filt.Label.isin(curated_ids)]["phi"].mean()
mean_pSmad_cur=feat_filt.loc[feat_filt.Label.isin(curated_ids)][stain].mean() #"pSmad1/5_nc_ratio"
cur_cells=feat_filt.loc[feat_filt.Label.isin(curated_ids)]
"""
fig0 = px.scatter_3d(cur_cells, x='x_corr', y='y_corr', z='z_corr', color="pSmad1/5_nc_ratio_mean_phi", opacity=1, color_continuous_scale="turbo")
fig0.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig2.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig0.write_html(path_out_im+"cur_cells_"+im+".html")
"""
print(mean_pSmad_cur)
cur_phi_0=mean_phi_cur
"""
if cur_phi_0>=0:
    cur_phi_0-=math.pi
else:
    cur_phi_0+=math.pi
"""
"""
cur_phi_sign=np.sign(cur_phi_0)
if cur_phi_sign==-1:
    feat_filt["cur_phi"]=np.abs(cur_phi_0)+feat_filt["phi"]
    feat_filt.loc[(feat_filt["cur_phi"]>math.pi), "cur_phi"]=feat_filt["cur_phi"]-2*math.pi
elif cur_phi_sign==1:
    feat_filt["cur_phi"]=feat_filt["phi"]-cur_phi_0
    feat_filt.loc[(feat_filt["cur_phi"]<-1*math.pi), "cur_phi"]=feat_filt["cur_phi"]+2*math.pi
else:
    feat_filt["cur_phi"]=feat_filt["phi"]
"""

print("find curated phi mean --- %s seconds ---\n" % (time.time() - start_time_1))
y_min=feat_filt[stain].quantile(0.01)
y_max=feat_filt[stain].quantile(0.99)
#feat_filt=feat_filt.loc[feat_filt[stain]>1]
feat_filt["rel_theta"]=feat_filt.theta/np.max(feat_filt.theta)
if stain=="betaCatenin_nuc":
    margin_loc=feat_filt.rel_theta.quantile(margin)
    feat_filt=feat_filt.loc[feat_filt.rel_theta>margin_loc].copy()

phi_shift_min={}
for shift in shifts:


    new_phi_sign=np.sign(shift)

    if new_phi_sign==-1:
        feat_filt["phi_temp"]=np.abs(shift)+feat_filt["phi"]
        feat_filt.loc[(feat_filt["phi_temp"]>math.pi), "phi_temp"]=feat_filt["phi_temp"]-2*math.pi
    elif new_phi_sign==1:
        feat_filt["phi_temp"]=feat_filt["phi"]-shift
        feat_filt.loc[(feat_filt["phi_temp"]<-1*math.pi), "phi_temp"]=feat_filt["phi_temp"]+2*math.pi
    else:
        feat_filt["phi_temp"]=feat_filt["phi"]
    
    feat_filt["phi_temp"]=np.abs(feat_filt["phi_temp"])
    phi_bins=np.linspace(0, math.pi, n_bins)
    labels = range(1, n_bins)
    
    feat_filt['phi_bin_temp'] = pd.to_numeric(pd.cut(x = feat_filt['phi_temp'], bins = phi_bins, labels = labels, include_lowest = True, right=False))

    phi_bins=phi_bins[:-1]
    means_ov_phi=list(feat_filt.groupby(['phi_bin_temp'])[stain].mean())

    shift_list=len(means_ov_phi)-1-means_ov_phi.index(max(means_ov_phi))
    means_ov_phi=np.roll(means_ov_phi, shift_list)
    #phi_bins=np.roll(phi_bins, shift_list)


    model = np.poly1d(np.polyfit(phi_bins, means_ov_phi, 2))
    polyline = np.linspace(min(phi_bins), max(phi_bins), 100)
    new_phi_min=np.poly1d.deriv(model).roots
    phi_shift_min[shift]=new_phi_min

"""
phi_bins=np.linspace(0, math.pi, n_bins)
labels = range(1, n_bins)
feat_filt.phi=np.abs(feat_filt.phi)
feat_filt['phi_bin_fit'] = pd.to_numeric(pd.cut(x = feat_filt['phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
phi_bins=phi_bins[:-1]
means_ov_phi=list(feat_filt.groupby(['phi_bin_fit'])["pSmad1/5_nc_ratio"].mean())

shift_list=len(means_ov_phi)-means_ov_phi.index(min(means_ov_phi))
means_ov_phi=np.roll(means_ov_phi, shift_list)
#phi_bins=np.roll(phi_bins, shift_list)


model = np.poly1d(np.polyfit(phi_bins, means_ov_phi, 2))
polyline = np.linspace(min(phi_bins), max(phi_bins), 100)
new_phi_min=np.poly1d.deriv(model).roots

fig1=plt.figure(1)
plt.scatter(phi_bins, means_ov_phi)
plt.plot(polyline, model(polyline))


print(new_phi_min)
print(cur_phi_0)
print(new_phi_min-cur_phi_0)
print((new_phi_min-cur_phi_0)*180/math.pi)
"""
fig1=plt.figure(1)
plt.scatter(phi_shift_min.keys(), phi_shift_min.values())
print(cur_phi_0)
plt.show()






