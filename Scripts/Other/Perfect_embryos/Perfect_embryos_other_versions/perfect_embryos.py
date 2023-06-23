#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:27:55 2023

@author: floriancurvaia
"""
## TODO: Use different embryos + noise
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.utils import shuffle
from random import sample


def high_dorsal(phi, a=0.35, k=20):
    y=1+1/(1+np.exp(k*(phi-a)))
    return(y)

def low_dorsal(phi, a=0.35, k=20):
    y=1+1/(1+np.exp(k*(phi-a)))
    y=1-y+2
    return(y)

def high_ventral(phi, a=2.75, k=20):
    y=1+1/(1+np.exp(k*(phi-a)))
    y=1-y+1
    return(y)

def high_dorsal_margin(phi, theta, a=0.35, k=20):
    y=1+1/(1+np.exp(k*(phi-a)))
    t_max=np.max(theta)
    y=y*theta/t_max
    return(y)

def low_dorsal_margin(phi, theta, a=0.35, k=15):
    y=1+1/(1+np.exp(k*(phi-a)))
    y=1-y+1
    #y=phi/np.max(phi)
    y=y*high_margin(theta, a=1.15, k=10)
    #t_max=np.max(theta)
    #y=y*theta/t_max
    return(y)

def high_ventral_margin(phi, theta, a=2.75, k=15):
    y=1+1/(1+np.exp(k*(phi-a)))
    y=1-y+1
    #y_t=1/(1+np.exp(-k_t*(theta-a_t)))
    #y=y*y_t
    #t_max=np.max(theta)
    #y=y*np.exp(theta/t_max)
    #y=phi/np.max(phi)
    y=y*high_margin(theta, a=1.15, k=10)
    return(y)

def high_margin(theta, a=1.15, k=10):
    y=1+1/(1+np.exp(k*(theta-a)))
    y=1-y
    #y=theta/np.max(theta)
    return(y)

def uniform(theta, m=0, std=1.e-2):
    n=len(theta)
    y=np.random.normal(m,std,n)
    #y+=1
    return(y)



#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"
#path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Perfect_embryos/"
path_out_csv="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/Perfect_embryos/"

path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Perfect_embryos/"
#path_out_csv="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/Perfect_embryos/"

"""

emb="B07_px+1257_py-0474"
fn=path_in+emb+"_w_dist_sph_simp.csv"
feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["theta", "dist_out", "cur_phi", "phi_bin_cur", "x_corr", "y_corr", "z_corr"])
feat_filt["phi_abs"]=feat_filt.cur_phi.abs()
#feat_filt=feat_filt.sort_values(["phi_abs"])
#feat_filt=feat_filt.sort_values(["theta"])
#tmax=feat_filt.theta.max()
"""
wells=["D06", "B07", "C07", "D07"]
fld=Path(path_in)
files=[]
all_df=[]
to_remove=["B07_px-0202_py+0631", "C07_px+0243_py-1998", "C07_px+0301_py+1212"]
for w in wells:
    for f in fld.glob(w+"*"+"_w_dist_sph_simp.csv"):
        name=f.name
        emb=name.split("_w")[0]
        if emb in to_remove:
            pass
        else:
            files.append(emb)
            fn=path_in+emb+"_w_dist_sph_simp.csv"
            feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["theta", "dist_out", "cur_phi", "phi_bin_cur", "x_corr", "y_corr", "z_corr"])
            feat_filt["emb"]=emb
            all_df.append(feat_filt)
            del feat_filt
        
#del files[9]
#del files[4]
#del files[3]
#del files[1]
#files.remove("C07_px+0243_py-1998")
#files.remove('B07_px-0202_py+0631')

feat_filt_all=pd.concat(all_df)
#prop_to_change=0.05

perf_stains=["low_dorsal_margin", "high_ventral_margin", "high_margin", "uniform"]
#perf_stains=["low_dorsal", "high_ventral", "high_margin", "uniform"]


for i in range(0, len(files)*3, 3):
    im=files[int(i/3)]
    for j in range(0,1):
        feat_filt_i=feat_filt_all.loc[feat_filt_all.emb==im].copy()
        feat_filt_i["rel_theta"]=feat_filt_i['theta']/np.max(feat_filt_i['theta'])
        feat_filt_i["phi_abs"]=feat_filt_i.cur_phi.abs()
        for func in perf_stains:
            noise=np.random.normal(1,1.e-1,len(feat_filt_i))
            #to_change=sample(range(len(feat_filt_i)), math.floor(len(feat_filt_i)*prop_to_change))
            
            if func=="high_margin" or func=="uniform":
                y=globals()[func](feat_filt_i.theta.to_numpy())
                #y[to_change]=shuffle(y)[to_change]
                y+=1
                #noise=np.apply_along_axis(np.random.normal, 0, y, scale=np.abs(y*0.3))
                feat_filt_i[func]=y+noise
            else:
                y=globals()[func](feat_filt_i.phi_abs.to_numpy(), feat_filt_i.theta.to_numpy())
                #y[to_change]=shuffle(y)[to_change]
                y+=1
                #noise=np.apply_along_axis(np.random.normal, 0, y, scale=np.abs(y*0.3))
                feat_filt_i[func]=y+noise
                
            fig_col_1 = px.scatter_3d(feat_filt_i, x='x_corr', y='y_corr', z='z_corr', color=func, opacity=1, color_continuous_scale="turbo", range_color=[feat_filt_i[func].min(), feat_filt_i[func].quantile(0.99)])
            fig_col_1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            fig_col_1.write_html(path_out_im+func+"_"+str(i+j)+".html")
            
        feat_filt_i.to_csv(path_out_csv+"Perfect_embryo_"+str(i+j)+".csv", sep=",", index=False)
    


    
    
    
    

"""
#feat_filt=feat_filt.loc[feat_filt.theta>1.175730422478265]
feat_filt=feat_filt_i
a=0.35
k=20
phi=np.linspace(0, math.pi, len(feat_filt)+1)
phi=phi[1:]
#y=a*x**-k+1
#y=1+1/(1+np.exp(k*(x-a)))

#y_noise=y+noise

n_ticks=15
phi_bins_abs=np.linspace(0, math.pi, n_ticks)
phi_labs_abs=360*phi_bins_abs/(2*math.pi)
theta_bins=np.linspace(0, 1, n_ticks)
theta_labs=360*theta_bins/(2*math.pi)
theta=np.linspace(0, 1, len(feat_filt)+1)
theta=theta[1:]

#phi_labs_abs=phi_labs_abs[:-1]



feat_filt=feat_filt_i
feat_filt.sort_values("phi_abs")
#y_emb=high_margin(theta, 0.7, 30)
y_emb=low_dorsal(phi)
#y_emb=uniform(theta)
noise=np.random.normal(0,10**-2,len(feat_filt))
y_emb_noise=y_emb+noise#+1

f, ax =plt.subplots(figsize=(10,10))
#ax.plot(phi,y_emb)
#ax.plot(theta,y_emb)
#ax.plot(theta, y_emb_noise)
#ax.set_xlabel("Theta")
ax.plot(phi, y_emb_noise)
ax.set_xlabel("Phi")
ax.set_ylabel("Intensity")
ax.set_title("Low dorsal")
ax.set_xticks(phi_bins_abs, np.round(phi_labs_abs, 1), rotation=90)
#ax.set_xticks(theta_bins, np.round(theta_labs, 1), rotation=90)
plt.show()
"""













