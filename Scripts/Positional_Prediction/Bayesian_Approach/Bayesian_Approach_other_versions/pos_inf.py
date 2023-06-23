#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:27:14 2023

@author: floriancurvaia
"""
from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

from scipy.stats import zscore

import scipy.stats as scst

path_in="/data/homes/fcurvaia/distances/"   #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/" #"/data/homes/fcurvaia/distances/" 
path_out_im="/data/homes/fcurvaia/Images/Pos_inf/"   #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Pos_inf/" #"/data/homes/fcurvaia/Images/Pos_inf/"


stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio', 'pSmad2/3_nc_ratio']
wells=["D06", "B07", "C07", "D07"]
fld=Path(path_in)
files=[]
all_df=[]
for w in wells:
    for f in fld.glob(w+"*"+"_w_dist_sph_simp.csv"):
        name=f.name
        emb=name.split("_w")[0]
        if emb == 'B07_px-0202_py+0631' or emb == "C07_px+0243_py-1998":
            pass
        else:
            files.append(emb)
            fn=path_in+emb+"_w_dist_sph_simp.csv"
            feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["r_neigh", "NTouchingNeighbors","PhysicalSize", "theta", "dist_out","phi", "phi_bin", "theta_bin", "cur_phi", "phi_bin_cur"]+stains)
            feat_filt["emb"]=emb
            all_df.append(feat_filt)
        
        
#files.remove("C07_px+0243_py-1998")
#files.remove('B07_px-0202_py+0631')

feat_filt_all=pd.concat(all_df)
#feat_filt_all=feat_filt_all.loc[feat_filt_all.emb !='B07_px-0202_py+0631']
#feat_filt_all=feat_filt_all.loc[feat_filt_all.emb !="C07_px+0243_py-1998"]





n_bins=20
phi_bins=np.linspace(-math.pi, math.pi, n_bins)
labels = range(1, n_bins)
theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)
phi_bins_abs=np.linspace(0, math.pi, n_bins)
phi_labs_abs=360*phi_bins_abs/(2*math.pi)
phi_labs_abs=phi_labs_abs[:-1]






for stain in stains:
    c=stain.split("_")
    stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
    
    #feat_filt_all[stain]=zscore(feat_filt_all[stain])
    
    all_means_phi=np.zeros((len(files), n_bins-1))
    all_means_theta=np.zeros((len(files), n_bins-1))
    all_means_all_phi=np.zeros((len(files), n_bins-1))
    all_means_all_theta=np.zeros((len(files), n_bins-1))
    
    f1, axs1 = plt.subplots(4,4, gridspec_kw={'width_ratios': [1, 1, 1, 1]}, figsize=(60, 60)) #, sharey=True)
    fon_size=10
    #space_size=38
    plt.rcParams.update({'font.size': fon_size}) 
    f1.subplots_adjust(hspace=0.125)
    f1.subplots_adjust(wspace=0.15)
    f1.set_dpi(300)
    
    f2, axs2 = plt.subplots(4,4, gridspec_kw={'width_ratios': [1, 1, 1, 1]}, figsize=(60, 60)) #, sharey=True)
    f2.subplots_adjust(hspace=0.125)
    f2.subplots_adjust(wspace=0.125)
    f2.set_dpi(300)
    

    for im, ax1, ax2, e in zip(files, axs1.flatten(), axs2.flatten(), range(len(files))):
        fn=path_in+im+"_w_dist_sph_simp.csv"
        #feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["r_neigh", "NTouchingNeighbors","PhysicalSize", "theta", "dist_out","phi", "phi_bin", "theta_bin", "cur_phi", "phi_bin_cur"]+stains)
        feat_filt=feat_filt_all.loc[feat_filt_all.emb==im]
        #feat_filt[stain]=zscore(feat_filt[stain])
        
        #feat_filt.betaCatenin_nuc=zscore(feat_filt.betaCatenin_nuc)
        
        feat_filt['phi_bin_pos'] = pd.to_numeric(pd.cut(x = feat_filt['phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
        feat_filt['theta_bin_pos'] = pd.to_numeric(pd.cut(x = feat_filt['theta'], bins = theta_bins, labels = labels, include_lowest = True, right=False))
        
        feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'].abs(), bins = phi_bins_abs, labels = labels, include_lowest = True, right=False))
        
        #feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
        t_max=np.max(feat_filt.theta_bin_pos)
        feat_filt_9=feat_filt.loc[feat_filt.theta_bin_pos==t_max-1]
        
        feat_filt_0=feat_filt.loc[feat_filt.phi_bin_abs==n_bins-1]
        
        #ax1.errorbar(np.unique(feat_filt_9.phi_bin_abs), feat_filt_9.groupby("phi_bin_abs")[stain].mean(), yerr=feat_filt_9.groupby("phi_bin_abs")[stain].std())
        
        #ax2.errorbar(np.unique(feat_filt_0.theta_bin_pos), feat_filt_0.groupby("theta_bin_pos")[stain].mean(), yerr=feat_filt_0.groupby("theta_bin_pos")[stain].std())
        
        ax1.errorbar(np.unique(feat_filt.phi_bin_abs), feat_filt.groupby("phi_bin_abs")[stain].mean(), yerr=feat_filt.groupby("phi_bin_abs")[stain].std())
        ax1.set_title(im)
        
        ax2.errorbar(np.unique(feat_filt.theta_bin_pos), feat_filt.groupby("theta_bin_pos")[stain].mean(), yerr=feat_filt.groupby("theta_bin_pos")[stain].std())
        ax2.set_title(im)
        
        phi_numpy=feat_filt_9.groupby("phi_bin_abs")[stain].mean().to_numpy().copy()
        phi_numpy.resize((n_bins-1), refcheck=False)
        phi_numpy[phi_numpy==0]=np.nan
        #all_means_phi[e]=zscore(phi_numpy, nan_policy="omit")
        all_means_phi[e]=phi_numpy
        
        theta_numpy=feat_filt_0.groupby("theta_bin_pos")[stain].mean().to_numpy().copy()
        theta_numpy.resize((n_bins-1), refcheck=False)
        theta_numpy[theta_numpy==0]=np.nan
        #all_means_theta[e]=zscore(theta_numpy, nan_policy="omit")
        all_means_theta[e]=theta_numpy
        
        
        all_phi_numpy=feat_filt.groupby("phi_bin_abs")[stain].mean().to_numpy().copy()
        all_phi_numpy.resize((n_bins-1), refcheck=False)
        all_phi_numpy[all_phi_numpy==0]=np.nan
        #all_means_all_phi[e]=zscore(all_phi_numpy, nan_policy="omit")
        all_means_all_phi[e]=all_phi_numpy
        
        all_theta_numpy=feat_filt.groupby("theta_bin_pos")[stain].mean().to_numpy().copy()
        all_theta_numpy.resize((n_bins-1), refcheck=False)
        all_theta_numpy[all_theta_numpy==0]=np.nan
        #all_means_all_theta[e]=zscore(all_theta_numpy, nan_policy="omit")
        all_means_all_theta[e]=all_theta_numpy
        
        
        
        
        
    f1.savefig(path_out_im+stain_clean+"_phi_plot.png", bbox_inches='tight', dpi=300)
    f2.savefig(path_out_im+stain_clean+"_theta_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
        
    
    #all_means_phi[all_means_phi == 0] = np.nan
    
    means_phi=np.nanmean(all_means_phi, axis=0)
    
    means_all_phi=np.nanmean(all_means_all_phi, axis=0)
    
    #all_means_theta[all_means_theta == 0] = np.nan
    
    means_theta=np.nanmean(all_means_theta, axis=0)
    
    means_all_theta=np.nanmean(all_means_all_theta, axis=0)
    
    std_phi=np.nanstd(all_means_phi, axis=0)
    
    std_all_phi=np.nanstd(all_means_all_phi, axis=0)
    
    std_theta=np.nanstd(all_means_theta, axis=0)
    
    std_all_theta=np.nanstd(all_means_all_theta, axis=0)
    
    delta_phi=(all_means_all_phi-means_all_phi)/std_all_phi
    delta_phi=delta_phi[~np.isnan(delta_phi)].flatten()
    
    shap_delta_phi_pval=scst.shapiro(delta_phi)[1]
    
    delta_theta=(all_means_all_theta-means_all_theta)/std_all_theta
    delta_theta=delta_theta[~np.isnan(delta_theta)].flatten()
    
    shap_delta_theta_pval=scst.shapiro(delta_theta)[1]
    
    phi_max=np.argmax(np.isnan(all_means_phi))
    if phi_max==0:
        phi_max=n_bins-1
    theta_max=np.argmax(np.isnan(all_means_theta))
    if theta_max==0:
        theta_max=n_bins-1
        
    phi_max_all=np.argmax(np.isnan(all_means_all_phi))
    if phi_max_all==0:
        phi_max_all=n_bins-1
    theta_max_all=np.argmax(np.isnan(all_means_all_theta))
    if theta_max_all==0:
        theta_max_all=n_bins-1
        
    fig, ax=plt.subplots()
    ax.errorbar(list(range(1, n_bins))[:phi_max], means_phi[:phi_max], yerr=std_phi[:phi_max])
    fig.savefig(path_out_im+stain_clean+"_means_phi_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    fig, ax=plt.subplots()
    ax.errorbar(list(range(1, n_bins))[:theta_max], means_theta[:theta_max], yerr=std_theta[:theta_max])
    fig.savefig(path_out_im+stain_clean+"_means_theta_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    fig, ax=plt.subplots()
    ax.errorbar(list(range(1, n_bins))[:phi_max_all], means_all_phi[:phi_max_all], yerr=std_all_phi[:phi_max_all])
    fig.savefig(path_out_im+stain_clean+"_means_all_phi_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    fig, ax=plt.subplots()
    ax.errorbar(list(range(1, n_bins))[:theta_max_all], means_all_theta[:theta_max_all], yerr=std_all_theta[:theta_max_all])
    fig.savefig(path_out_im+stain_clean+"_means_all_theta_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    x_norm = np.linspace(-4, 4,100)
    
    fig, ax=plt.subplots()
    ax.hist(delta_phi, bins=25, density=True)
    plt.plot(x_norm,scst.norm.pdf(x_norm,0,1))
    ax.set_title(shap_delta_phi_pval)
    fig.savefig(path_out_im+stain_clean+"_delta_phi.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    fig, ax=plt.subplots()
    ax.hist(delta_theta, bins=25, density=True)
    plt.plot(x_norm,scst.norm.pdf(x_norm,0,1))
    ax.set_title(shap_delta_theta_pval)
    fig.savefig(path_out_im+stain_clean+"_delta_theta.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    