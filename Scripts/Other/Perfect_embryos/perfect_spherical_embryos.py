#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:10:31 2023

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


#From: https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return np.array(vec)

def sample_phi_theta(npoints):
    phi=np.random.rand(10000)*2*math.pi
    theta=np.random.rand(10000)*math.pi/2
    return(np.array([phi, theta]))

def appendSpherical_pd_ed(df): #from https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    xy = df["x_corr"]**2+df["y_corr"]**2
    df["r"] = np.sqrt(xy + df["z_corr"]**2)
    df["theta"]= np.arccos(df["z_corr"]/df["r"])
    df["phi"] = np.sign(df["y_corr"])*np.arccos(df["x_corr"]/np.sqrt(xy))
    return df

def appendCartesian(df, r=350): #from https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    df["x_corr"]=np.sin(df.theta)*np.cos(df.phi)*r
    df["y_corr"]=np.sin(df.theta)*np.sin(df.phi)*r
    df["z_corr"]=np.cos(df.theta)*r
    return df

def high_dorsal(phi, a=0.35, k=20):
    y=1+1/(1+np.exp(k*(phi-a)))
    return(y)

def low_dorsal(phi, a=0.35, k=15):
    y=1+1/(1+np.exp(k*(phi-a)))
    y=1-y+1
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
    y=y*high_margin(theta, a=1.4, k=15)
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
    y=y*high_margin(theta, a=1.4, k=15)
    return(y)

def high_margin(theta, a=1.4, k=15):
    y=1+1/(1+np.exp(k*(theta-a)))
    y=1-y+1
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

n_emb=50

n_bins=20
labels = range(1, n_bins)
#theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)
theta_bins=np.linspace(0, math.pi/2, n_bins, endpoint=True)
phi_bins_abs=np.linspace(0, math.pi, n_bins)
phi_labs_abs=360*phi_bins_abs/(2*math.pi)
phi_labs_abs=phi_labs_abs[:-1]

"""

emb="B07_px+1257_py-0474"
fn=path_in+emb+"_w_dist_sph_simp.csv"
feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["theta", "dist_out", "cur_phi", "phi_bin_cur", "x_corr", "y_corr", "z_corr"])
feat_filt["phi_abs"]=feat_filt.cur_phi.abs()
#feat_filt=feat_filt.sort_values(["phi_abs"])
#feat_filt=feat_filt.sort_values(["theta"])
#tmax=feat_filt.theta.max()
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

"""
perf_stains=["low_dorsal_margin", "high_ventral_margin", "high_margin", "uniform"]
#perf_stains=["low_dorsal", "high_ventral", "high_margin", "uniform"]

#x_i, y_i, z_i=sample_spherical(20000, 3)

feat_filt=pd.DataFrame()
feat_filt[["x_corr", "y_corr", "z_corr"]]=sample_spherical(20000, 3).T*350
feat_filt=feat_filt.loc[feat_filt.z_corr>=0]
#feat_filt[["phi", "theta"]]=sample_phi_theta(10000).T
feat_filt=appendSpherical_pd_ed(feat_filt)
#feat_filt=appendCartesian(feat_filt)
#feat_filt["phi"]-=math.pi
feat_filt["phi_abs"]=feat_filt.phi.abs()
feat_filt['theta_bin'] = pd.to_numeric(pd.cut(x = feat_filt['theta'], bins = theta_bins, labels = labels, include_lowest = True, right=True)) #/np.max(feat_filt_all['theta'])
feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['phi_abs'], bins = phi_bins_abs, labels = labels, include_lowest = True, right=False))




for i in range(n_emb):
    feat_filt_i=feat_filt.copy()
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
        fig_col_1.update_traces(marker=dict(line=dict(width=10,color='black')))
        fig_col_1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig_col_1.write_html(path_out_im+func+"_"+str(i)+".html")
        
    feat_filt_i.to_csv(path_out_csv+"Perfect_embryo_"+str(i)+".csv", sep=",", index=False)
    

"""
    
    
    


#feat_filt=feat_filt.loc[feat_filt.theta>1.175730422478265]
a=0.35
k=20
phi=np.linspace(0, math.pi, len(feat_filt)+1)
phi=phi[1:]
#y=a*x**-k+1
#y=1+1/(1+np.exp(k*(x-a)))
noise=np.random.normal(0,10**-2,len(feat_filt))
#y_noise=y+noise

n_ticks=15
phi_bins_abs=np.linspace(0, math.pi, n_ticks)
phi_labs_abs=360*phi_bins_abs/(2*math.pi)
theta_bins=np.linspace(0, math.pi/2, n_ticks)
theta_labs=360*theta_bins/(2*math.pi)
theta=np.linspace(0, math.pi/2, len(feat_filt)+1)
theta=theta[1:]

#phi_labs_abs=phi_labs_abs[:-1]
i="null"
feat_filt["low_dorsal"]=low_dorsal(feat_filt.phi_abs)
func="low_dorsal"
fig_col_1 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="low_dorsal", opacity=1, color_continuous_scale="turbo", range_color=[feat_filt[func].min(), feat_filt[func].quantile(0.99)])
fig_col_1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig_col_1.write_html(path_out_im+func+"_"+str(i)+".html")
    
#feat_filt.to_csv(path_out_csv+"Perfect_embryo_"+str(i)+".csv", sep=",", index=False)

feat_filt.sort_values("theta", inplace=True)
y_emb=high_margin(feat_filt.theta.to_numpy(), 1.4, 15)
#y_emb=high_ventral(phi)
y_emb_noise=y_emb+noise

f, ax =plt.subplots()
#ax.plot(phi,y_emb_noise)
ax.plot(theta,y_emb_noise)
#ax.plot(theta, y_emb_noise)
#ax.set_xticks(phi_bins_abs, np.round(phi_labs_abs, 1), rotation=90)
ax.set_xticks(theta_bins, np.round(theta_labs, 1), rotation=90)

plt.show()


"""










