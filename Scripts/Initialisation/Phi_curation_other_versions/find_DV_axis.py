#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:50:16 2023

@author: floriancurvaia
"""
import numpy as np

import time

import pandas as pd

import math

import plotly.express as px



start_time_0=time.time()

im="B02_px-0280_py+1419" 
#B02_px-0280_py+1419, B03_px-0545_py-1946, B04_px+0114_py-1436, B05_px+1522_py-1087, C05_px+0198_py+1683, 
#D05_px-0738_py-1248, B06_px-0493_py+0146, D06_px-1055_py-0118, B07_px+1257_py-0474, B08_px+1076_py-0189
fn_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv" #B08_px+1076_py-0189_w_dist_sph_simp.csv"
feat_filt=pd.read_csv(fn_dist, sep=",", index_col=False)
n_bins=30
print("load files --- %s seconds ---\n" % (time.time() - start_time_0))


start_time_1=time.time()
means_psmad15_phi_theta=feat_filt.groupby(['phi_bin', 'theta_bin'])["pSmad1/5_nc_ratio"].mean()

feat_filt["pSmad1/5_nc_ratio_mean_phi_theta"]=0

for i in range(1, n_bins):
    for j in range(1, n_bins):
        if (i,j) in means_psmad15_phi_theta.index:
            #print("yes")
            feat_filt.loc[(feat_filt["phi_bin"]==i) & (feat_filt["theta_bin"]==j), "pSmad1/5_nc_ratio_mean_phi_theta"]=means_psmad15_phi_theta.loc[(i,j)]#math.log(means_psmad15_phi_theta.loc[(i,j)])
            
            

print("find mean per bin phi theta --- %s seconds ---\n" % (time.time() - start_time_1))


start_time_2=time.time()
means_psmad15_phi=feat_filt.groupby(['phi_bin'])["pSmad1/5_nc_ratio"].mean()

feat_filt["pSmad1/5_nc_ratio_mean_phi"]=0

for i in range(1, n_bins):
    if i in means_psmad15_phi.index:
        feat_filt.loc[(feat_filt["phi_bin"]==i), "pSmad1/5_nc_ratio_mean_phi"]=means_psmad15_phi.loc[(i)]#math.log(means_psmad15_phi.loc[(i)])
            
            

print("find mean per bin phi --- %s seconds ---\n" % (time.time() - start_time_2))

phi_bins=np.linspace(-math.pi, math.pi, n_bins)
bin_max=means_psmad15_phi.loc[means_psmad15_phi==max(means_psmad15_phi)].index[0]
bin_range=[phi_bins[bin_max-1], phi_bins[bin_max]]
new_phi_0=np.mean(bin_range)

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






path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/DV_axis/"

fig1 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="pSmad1/5_nc_ratio_mean_phi_theta", opacity=1, color_continuous_scale="turbo")
fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig1.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig1.write_html(path_out_im+"pSmad15_nc_ratio_mean_phi_theta_"+im+".html")




fig2 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="pSmad1/5_nc_ratio_mean_phi", opacity=1, color_continuous_scale="turbo")
fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig2.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig2.write_html(path_out_im+"pSmad15_nc_ratio_mean_phi_"+im+".html")


feat_filt["pSmad1/5_nc_ratio_log"]=np.log(feat_filt["pSmad1/5_nc_ratio"])

fig3 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="pSmad1/5_nc_ratio_log", opacity=1, color_continuous_scale="turbo")
fig3.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig3.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig3.write_html(path_out_im+"pSmad15_nc_ratio_log_"+im+".html")


fig4 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="new_phi", opacity=1, color_continuous_scale="turbo")
fig4.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig3.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig4.write_html(path_out_im+"new_phi_"+im+".html")









