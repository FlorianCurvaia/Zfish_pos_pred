#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:26:16 2023

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

plt.ioff()

stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio', 'pSmad2/3_nc_ratio']
start_time_0=time.time()



path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/3D_scatter/"
#path_out_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"
path_out_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
images=['D06_px-0377_py-1358', 'D06_px+1741_py-0131', 'D06_px+0217_py+1735', 'D06_px-1055_py-0118', 
        'B07_px-0202_py+0631', 'B07_px+1257_py-0474', 'B07_px+0030_py-1959', 'B07_px-2056_py-0041', 
        'B07_px+0282_py+1729', 'C07_px+1425_py+0902', 'C07_px+1929_py-0176', 'C07_px-0784_py-0616', 
        'C07_px+0243_py-1998', 'C07_px+0301_py+1212', 'D07_px-0500_py+0670', 'D07_px+0999_py-1281']

images_1=["B02_px-0280_py+1419", "B03_px-0545_py-1946", "C04_px-0816_py-1668", "B05_px+1522_py-1087", "C05_px+0198_py+1683", 
        "D05_px-0738_py-1248", "B06_px-0493_py+0146", "D06_px-1055_py-0118", "B07_px+1257_py-0474", "B08_px+1076_py-0189"]
images+=images_1

to_remove=['D06_px+1741_py-0131', 'D06_px-1055_py-0118', 'B07_px+1257_py-0474', 'C07_px+1929_py-0176', "B07_px-0202_py+0631", "C07_px+0243_py-1998"]

images.remove('B07_px-0202_py+0631')
images.remove("C07_px+0243_py-1998")
#images=list(set(images)-set(to_remove))

time_emb={"B02":3.3, "C02":3.3, "D02":3.3, "B03":3.7, "C03":3.7, "D03":3.7, "B04":4.3, "C04":4.3, "D04":4.3,
          "B05":4.7, "C05":5.3, "D05":5.7, "B06":5.7, "C06":5.7, "D06":6.0, "B07":6.0, "C07":6.0, "D07":6.0,
          "B08":7.0, "C08":7.0, "D08":7.0}


margin=0.8

n_bins=18
labels = range(1, n_bins)
#theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)
theta_bins=np.linspace(0, 1, n_bins, endpoint=True)
phi_bins_abs=np.linspace(0, math.pi, n_bins)
phi_labs_abs=360*phi_bins_abs/(2*math.pi)
phi_labs_abs=phi_labs_abs[:-1]


for idx in range(len(images)):
    im=images[idx]
    fn_dist=path_out_dist+im+"_w_dist_sph_simp.csv" #B08_px+1076_py-0189_w_dist_sph_simp.csv"
    feat_filt=pd.read_csv(fn_dist, sep=",", index_col=False)
    t_max=feat_filt.theta.max()
    feat_filt["rel_theta"]=feat_filt.theta/t_max
    feat_filt["cur_phi_abs"]=feat_filt.cur_phi.abs()
    well=im.split("_")[0]
    theta_labs=theta_bins*t_max*360/(2*math.pi)
    
    for stain in stains:
        c=stain.split("_")
        stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
        
        y_min=feat_filt[stain].quantile(0.01)
        y_max=feat_filt[stain].quantile(0.99)
        feat_filt=feat_filt.loc[feat_filt[stain]<y_max]
        feat_filt=feat_filt.loc[feat_filt[stain]>y_min]
        #feat_filt[stain]=stats.zscore(feat_filt[stain])
        #feat_filt=feat_filt.loc[feat_filt[stain]>1]
        
        fig1 = px.scatter_3d(feat_filt, x="cur_phi_abs", y="rel_theta", z=stain, color=stain, opacity=0.75, color_continuous_scale="turbo", range_color=[y_min, y_max])
        #fig1.update_traces(marker=dict(line=dict(width=10,color='black')))
        fig1.update_scenes(aspectmode="cube")
        fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig1.write_html(path_out_im+im+"_"+stain_clean+"_phi_abs_rel_theta.html")
        
        fig2 = px.scatter_3d(feat_filt, x="cur_phi", y="rel_theta", z=stain, color=stain, opacity=0.75, color_continuous_scale="turbo", range_color=[y_min, y_max])
        #fig2.update_traces(marker=dict(line=dict(width=10,color='black')))
        fig2.update_scenes(aspectmode="cube")
        fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig2.write_html(path_out_im+im+"_"+stain_clean+"_phi_rel_theta.html")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    