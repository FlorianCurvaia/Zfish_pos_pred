#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:45:24 2023

@author: floriancurvaia
"""

import numpy as np

import time

import pandas as pd

import math

import seaborn as sns

import matplotlib.pyplot as plt

start_time_0=time.time()
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/heatmaps/"
im="B07_px+1257_py-0474" 
#B02_px-0280_py+1419, B03_px-0545_py-1946, B04_px+0114_py-1436, B05_px+1522_py-1087, C05_px+0198_py+1683, 
#D05_px-0738_py-1248, B06_px-0493_py+0146, D06_px-1055_py-0118, B07_px+1257_py-0474, B08_px+1076_py-0189
fn_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv" #B08_px+1076_py-0189_w_dist_sph_simp.csv"
stain="pSmad1/5_nc_ratio"
n_bins=30
feat_filt=pd.read_csv(fn_dist, sep=",", index_col=False)

phi_bins=np.linspace(-math.pi, math.pi, n_bins)
theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)

t_max=np.max(feat_filt.theta_bin)

theta_bins=theta_bins[:t_max]
phi_bins=phi_bins#[:-1]

theta_labs=360*theta_bins/(2*math.pi)
theta_labs=theta_labs#[:t_max]
phi_labs=360*phi_bins/(2*math.pi)
phi_labs=phi_labs

fig0=plt.figure(0)
plt.clf()
vmin=feat_filt[stain].quantile(0.05)
vmax=feat_filt[stain].quantile(0.95)
ax=plt.scatter(feat_filt.theta, feat_filt.new_phi, c=feat_filt[stain], marker="o", s=8, vmin=vmin, vmax=vmax, cmap="jet")
plt.colorbar(ax, label=stain+" intensity")
plt.xlabel("theta")
plt.ylabel("phi")
plt.xticks(theta_bins, np.around(theta_labs, 2))
plt.yticks(phi_bins, np.around(phi_labs, 2))
#fig0.savefig(path_out_im+im+"_"+stain_clean+"_phi_new_theta_sing_cell.png", bbox_inches='tight', dpi=300)
#plt.close(fig0)

















