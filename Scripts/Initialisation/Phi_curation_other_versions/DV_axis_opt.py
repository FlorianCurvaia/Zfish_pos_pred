#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:10:53 2023

@author: floriancurvaia
"""


import numpy as np

import time

import pandas as pd

import math

import plotly.express as px

import matplotlib.pyplot as plt

start_time_0=time.time()
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/DV_axis/Opt/"
path_out_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"
im="B08_px+1076_py-0189" 
#B02_px-0280_py+1419, B03_px-0545_py-1946, B04_px+0114_py-1436, B05_px+1522_py-1087, C05_px+0198_py+1683, 
#D05_px-0738_py-1248, B06_px-0493_py+0146, D06_px-1055_py-0118, B07_px+1257_py-0474, B08_px+1076_py-0189
fn_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv" #B08_px+1076_py-0189_w_dist_sph_simp.csv"
feat_filt=pd.read_csv(fn_dist, sep=",", index_col=False)
n_bins=60
shifts=np.linspace(-math.pi, math.pi, 63)
print("load files --- %s seconds ---\n" % (time.time() - start_time_0))


start_time_1=time.time()
curated_ids=[5756,6708,6455,6282,6366,5830] #B08
#curated_ids=[6485,6919,7266,6300,6799,6838] #B07
#curated_ids=[5212, 5056, 5174, 4950, 5422, 4407] #C05
mean_phi_cur=feat_filt.loc[feat_filt.Label.isin(curated_ids)]["phi"].mean()
mean_pSmad_cur=feat_filt.loc[feat_filt.Label.isin(curated_ids)]["pSmad1/5_nc_ratio"].mean()
cur_cells=feat_filt.loc[feat_filt.Label.isin(curated_ids)]
fig0 = px.scatter_3d(cur_cells, x='x_corr', y='y_corr', z='z_corr', color="pSmad1/5_nc_ratio_mean_phi", opacity=1, color_continuous_scale="turbo")
fig0.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig2.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig0.write_html(path_out_im+"cur_cells_"+im+".html")

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

min_bins={}
min_bins_diff={}
min_bins_2={}
start_time_2=time.time()
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
        
    phi_bins=np.linspace(-math.pi, math.pi, n_bins)
    labels = range(1, n_bins)
    theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)
    
    feat_filt['phi_bin_DV'] = pd.to_numeric(pd.cut(x = feat_filt['phi_temp'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
    
    
    
    means_psmad15_phi=feat_filt.groupby(['phi_bin_DV'])["pSmad1/5_nc_ratio"].mean() #"pSmad1/5_nc_ratio" #"betaCatenin_nuc"

    
    feat_filt["pSmad1/5_nc_ratio_mean_phi"]=0
    
    for i in range(1, n_bins):
        if i in means_psmad15_phi.index:
            feat_filt.loc[(feat_filt["phi_bin_DV"]==i), "pSmad1/5_nc_ratio_mean_phi"]=means_psmad15_phi.loc[(i)]#math.log(means_psmad15_phi.loc[(i)])
                
                
    bin_min=means_psmad15_phi.loc[means_psmad15_phi==min(means_psmad15_phi)].index[0]
    bin_range=[phi_bins[bin_min-1], phi_bins[bin_min]]
    new_phi_0=feat_filt.loc[feat_filt["pSmad1/5_nc_ratio_mean_phi"]==min(means_psmad15_phi), "phi"].mean()#np.mean(bin_range)
    min_bins[new_phi_0]=min(means_psmad15_phi)
    min_bins_diff[new_phi_0]=new_phi_0-cur_phi_0
    min_bins_2[new_phi_0]=bin_min
    
    """
    min_clean="".join(str(min(means_psmad15_phi)).split("."))
    fig2 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="pSmad1/5_nc_ratio_mean_phi", opacity=1, color_continuous_scale="turbo")
    fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    #fig2.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
    fig2.write_html(path_out_im+"pSmad15_nc_ratio_mean_phi_"+im+"_"+min_clean+".html")
    """

print("find mean per bin phi --- %s seconds ---\n" % (time.time() - start_time_2))

fig1=plt.figure(1)
plt.scatter(min_bins.keys(), min_bins.values())

fig2=plt.figure(2)
plt.scatter(min_bins_diff.keys(), min_bins_diff.values())

fig3=plt.figure(3)
plt.scatter(min_bins_diff.values(), min_bins.values())

fig4=plt.figure(4)
plt.scatter(min_bins_2.keys(), min_bins_2.values())

"""
new_phi_sign=np.sign(new_phi_0)

if new_phi_sign==-1:
    feat_filt["new_phi"]=np.abs(new_phi_0)+feat_filt["phi"]
    feat_filt.loc[(feat_filt["new_phi"]>math.pi), "new_phi"]=feat_filt["new_phi"]-2*math.pi
elif new_phi_sign==1:
    feat_filt["new_phi"]=feat_filt["phi"]-new_phi_0
    feat_filt.loc[(feat_filt["new_phi"]<-1*math.pi), "new_phi"]=feat_filt["new_phi"]+2*math.pi
else:
    feat_filt["new_phi"]=feat_filt["phi"]

#feat_filt.loc[(feat_filt["new_phi"]<-1*math.pi) | (feat_filt["new_phi"]>math.pi), "new_phi"]=feat_filt.loc[(feat_filt["new_phi"]<-1*math.pi) | (feat_filt["new_phi"]>math.pi), "new_phi"] *-1

phi_bins=np.linspace(-math.pi, math.pi, 30)
labels = range(1, 30)

feat_filt['phi_bin_new'] = pd.to_numeric(pd.cut(x = feat_filt['new_phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
feat_filt['phi_bin_cur'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))

#feat_filt.to_csv(path_out_dist+im+"_w_dist_sph_simp.csv", sep=",")
"""
print(new_phi_0-cur_phi_0)


"""
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/DV_axis/Check/"

fig1 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="cur_phi", opacity=1, color_continuous_scale="turbo")
fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig1.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig1.write_html(path_out_im+"cur_phi_"+im+".html")




fig2 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="pSmad1/5_nc_ratio_mean_phi", opacity=1, color_continuous_scale="turbo")
fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig2.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig2.write_html(path_out_im+"pSmad15_nc_ratio_mean_phi_DV_"+im+".html")

feat_filt["pSmad1/5_nc_ratio_log"]=np.log(feat_filt["pSmad1/5_nc_ratio"])


fig3 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="pSmad1/5_nc_ratio_log", opacity=1, color_continuous_scale="turbo")
fig3.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig3.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig3.write_html(path_out_im+"pSmad15_nc_ratio_log_"+im+".html")


fig4 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="new_phi", opacity=1, color_continuous_scale="turbo")
fig4.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig3.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig4.write_html(path_out_im+"new_phi_DV_"+im+".html")


feat_filt.sort_values("new_phi", inplace=True)
plt.figure(1)
plt.clf()
plt.scatter(feat_filt.new_phi, feat_filt["pSmad1/5_nc_ratio"])#, c=feat_filt.KMeans_3_c) #, s=(feat_filt.KMeans_3_c+1)*100, c=feat_filt.betaCatenin_cyto)

feat_filt.sort_values("cur_phi", inplace=True)
plt.figure(2)
plt.clf()
plt.scatter(feat_filt.cur_phi, feat_filt["pSmad1/5_nc_ratio"])#, c=feat_filt.KMeans_3_c) #, s=(feat_filt.KMeans_3_c+1)*100, c=feat_filt.betaCatenin_cyto)

"""





