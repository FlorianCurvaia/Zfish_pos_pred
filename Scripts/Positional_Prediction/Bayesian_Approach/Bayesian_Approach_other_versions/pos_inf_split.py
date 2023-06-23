#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:34:02 2023

@author: floriancurvaia
"""

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

from scipy.stats import zscore

import scipy.stats as scst

#path_in="/data/homes/fcurvaia/distances/"   #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/" #"/data/homes/fcurvaia/distances/" 
#path_out_im="/data/homes/fcurvaia/Images/Pos_inf/"   #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Pos_inf/" #"/data/homes/fcurvaia/Images/Pos_inf/"

path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Pos_inf/"

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
            del feat_filt
        
        
#files.remove("C07_px+0243_py-1998")
#files.remove('B07_px-0202_py+0631')

feat_filt_all=pd.concat(all_df)
#feat_filt_all=feat_filt_all.loc[feat_filt_all.emb !='B07_px-0202_py+0631']
#feat_filt_all=feat_filt_all.loc[feat_filt_all.emb !="C07_px+0243_py-1998"]



hist_bins=20

n_bins=36
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
    
    #all_means_phi=np.zeros((len(files)*2, n_bins-1))
    #all_means_theta=np.zeros((len(files)*2, n_bins-1))
    all_means_all_phi=np.zeros((len(files)*2, n_bins-1))
    all_means_all_theta=np.zeros((len(files)*2, n_bins-1))
    
    """
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
    """
    

    #for im, ax1, ax2, e in zip(files, axs1.flatten(), axs2.flatten(), range(0,len(files)*2, 2)):
    for im, e in zip(files, range(0,len(files)*2, 2)):
        fn=path_in+im+"_w_dist_sph_simp.csv"
        #feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["r_neigh", "NTouchingNeighbors","PhysicalSize", "theta", "dist_out","phi", "phi_bin", "theta_bin", "cur_phi", "phi_bin_cur"]+stains)
        feat_filt=feat_filt_all.loc[feat_filt_all.emb==im]
        y_max=feat_filt[stain].quantile(0.975)
        y_min=feat_filt[stain].quantile(0.025)
        feat_filt=feat_filt.loc[feat_filt[stain]>y_min]
        feat_filt=feat_filt.loc[feat_filt[stain]<y_max]
        #feat_filt[stain]=zscore(feat_filt[stain])
        
        feat_filt_p=feat_filt.loc[feat_filt.cur_phi>=0]
        feat_filt_m=feat_filt.loc[feat_filt.cur_phi<0]
        
        
        #feat_filt.betaCatenin_nuc=zscore(feat_filt.betaCatenin_nuc)
        
        #feat_filt['phi_bin_pos'] = pd.to_numeric(pd.cut(x = feat_filt['phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
        feat_filt['theta_bin_pos'] = pd.to_numeric(pd.cut(x = feat_filt['theta'], bins = theta_bins, labels = labels, include_lowest = True, right=False))
        feat_filt_p['theta_bin_pos'] = pd.to_numeric(pd.cut(x = feat_filt_p['theta'], bins = theta_bins, labels = labels, include_lowest = True, right=False))
        feat_filt_m['theta_bin_pos'] = pd.to_numeric(pd.cut(x = feat_filt_m['theta'], bins = theta_bins, labels = labels, include_lowest = True, right=False))
        
        
        feat_filt_m['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt_m['cur_phi'].abs(), bins = phi_bins_abs, labels = labels, include_lowest = True, right=False))
        feat_filt_p['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt_p['cur_phi'].abs(), bins = phi_bins_abs, labels = labels, include_lowest = True, right=False))
        
        #feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
        t_max=np.max(feat_filt.theta_bin_pos)
        del feat_filt
        
        #ax1.errorbar(np.unique(feat_filt_p.phi_bin_abs), feat_filt_p.groupby("phi_bin_abs")[stain].mean(), yerr=feat_filt_p.groupby("phi_bin_abs")[stain].std())
        
        #ax2.errorbar(np.unique(feat_filt_p.theta_bin_pos), feat_filt_p.groupby("theta_bin_pos")[stain].mean(), yerr=feat_filt_p.groupby("theta_bin_pos")[stain].std())
        
        
        all_phi_numpy=feat_filt_p.groupby("phi_bin_abs")[stain].mean().to_numpy().copy()
        all_phi_numpy.resize((n_bins-1), refcheck=False)
        all_phi_numpy[all_phi_numpy==0]=np.nan
        #all_means_all_phi[e]=zscore(all_phi_numpy, nan_policy="omit")
        all_means_all_phi[e]=all_phi_numpy
        
        all_phi_numpy=feat_filt_m.groupby("phi_bin_abs")[stain].mean().to_numpy().copy()
        all_phi_numpy.resize((n_bins-1), refcheck=False)
        all_phi_numpy[all_phi_numpy==0]=np.nan
        #all_means_all_phi[e]=zscore(all_phi_numpy, nan_policy="omit")
        all_means_all_phi[e+1]=all_phi_numpy
        
        all_theta_numpy=feat_filt_p.groupby("theta_bin_pos")[stain].mean().to_numpy().copy()
        all_theta_numpy.resize((n_bins-1), refcheck=False)
        all_theta_numpy[all_theta_numpy==0]=np.nan
        #all_means_all_theta[e]=zscore(all_theta_numpy, nan_policy="omit")
        all_means_all_theta[e]=all_theta_numpy
        
        all_theta_numpy=feat_filt_m.groupby("theta_bin_pos")[stain].mean().to_numpy().copy()
        all_theta_numpy.resize((n_bins-1), refcheck=False)
        all_theta_numpy[all_theta_numpy==0]=np.nan
        #all_means_all_theta[e]=zscore(all_theta_numpy, nan_policy="omit")
        all_means_all_theta[e+1]=all_theta_numpy
        
        
        
        
        
    #f1.savefig(path_out_im+stain_clean+"_phi_plot.png", bbox_inches='tight', dpi=300)
    #f2.savefig(path_out_im+stain_clean+"_theta_plot.png", bbox_inches='tight', dpi=300)
    #plt.close()
        
    
    #all_means_phi[all_means_phi == 0] = np.nan
    
    #means_phi=np.nanmean(all_means_phi, axis=0)
    
    means_all_phi=np.nanmean(all_means_all_phi, axis=0)
    
    #all_means_theta[all_means_theta == 0] = np.nan
    
    #means_theta=np.nanmean(all_means_theta, axis=0)
    
    means_all_theta=np.nanmean(all_means_all_theta, axis=0)
    
    i_min_phi=np.nanmin(means_all_phi)
    i_max_phi=np.nanmax(means_all_phi)
    
    i_min_theta=np.nanmin(means_all_theta)
    i_max_theta=np.nanmax(means_all_theta)
    
    all_means_all_phi=(all_means_all_phi-i_min_phi)/(i_max_phi-i_min_phi)
    
    all_means_all_theta=(all_means_all_theta-i_min_theta)/(i_max_theta-i_min_theta)
    
    means_all_phi=(means_all_phi-i_min_phi)/(i_max_phi-i_min_phi)
    
    means_all_theta=(means_all_theta-i_min_theta)/(i_max_theta-i_min_theta)
    
    #std_phi=np.nanstd(all_means_phi, axis=0)
    
    std_all_phi=np.nanstd(all_means_all_phi, axis=0)
    
    #std_theta=np.nanstd(all_means_theta, axis=0)
    
    std_all_theta=np.nanstd(all_means_all_theta, axis=0)
    
    delta_phi_hist=np.zeros((len(files)*2, hist_bins))
    
    delta_phi=(all_means_all_phi-means_all_phi)/std_all_phi
    
    delta_phi_no_nan=delta_phi[~np.isnan(delta_phi)].flatten()
    
    #shap_delta_phi_pval=scst.shapiro(delta_phi_no_nan)[1]
    kol_delta_phi_pval=scst.kstest(delta_phi_no_nan, scst.norm.cdf)[1]
    
    delta_theta_hist=np.zeros((len(files)*2, hist_bins))
    
    delta_theta=(all_means_all_theta-means_all_theta)/std_all_theta
    delta_theta_no_nan=delta_theta[~np.isnan(delta_theta)].flatten()
    
    #shap_delta_theta_pval=scst.shapiro(delta_theta_no_nan)[1]
    kol_delta_theta_pval=scst.kstest(delta_theta_no_nan, scst.norm.cdf)[1]
    
    for i in range(0,len(files)*2, 2):
        delta_phi_hist[i]=np.histogram(delta_phi[i], density=True, bins=hist_bins, range=(-4,4))[0]
        delta_theta_hist[i]=np.histogram(delta_theta[i], density=True, bins=hist_bins, range=(-4,4))[0]
        
        delta_phi_hist[i+1]=np.histogram(delta_phi[i+1], density=True, bins=hist_bins, range=(-4,4))[0]
        delta_theta_hist[i+1]=np.histogram(delta_theta[i+1], density=True, bins=hist_bins, range=(-4,4))[0]
        
    
    std_delta_phi_hist=np.nanstd(delta_phi_hist, axis=0)
    std_delta_theta_hist=np.nanstd(delta_theta_hist, axis=0)
    
    means_delta_phi_hist=np.nanmean(delta_phi_hist, axis=0)
    means_delta_theta_hist=np.nanmean(delta_theta_hist, axis=0)
    
    """
    phi_max=np.argmax(np.isnan(all_means_phi))
    if phi_max==0:
        phi_max=n_bins-1
    theta_max=np.argmax(np.isnan(all_means_theta))
    if theta_max==0:
        theta_max=n_bins-1
    """
    
    phi_max_all=np.argmax(np.isnan(all_means_all_phi))
    if phi_max_all==0:
        phi_max_all=n_bins-1
    theta_max_all=np.argmax(np.isnan(all_means_all_theta))
    if theta_max_all==0:
        theta_max_all=n_bins-1
    
    """
    fig, ax=plt.subplots()
    ax.errorbar(list(range(1, n_bins))[:phi_max], means_phi[:phi_max], yerr=std_phi[:phi_max])
    fig.savefig(path_out_im+stain_clean+"_means_phi_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    fig, ax=plt.subplots()
    ax.errorbar(list(range(1, n_bins))[:theta_max], means_theta[:theta_max], yerr=std_theta[:theta_max])
    fig.savefig(path_out_im+stain_clean+"_means_theta_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    """
    
    fig, ax=plt.subplots()
    ax.errorbar(list(range(1, n_bins))[:phi_max_all], means_all_phi[:phi_max_all], yerr=std_all_phi[:phi_max_all])
    fig.savefig(path_out_im+stain_clean+"_means_all_phi_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    fig, ax=plt.subplots()
    ax.errorbar(list(range(1, n_bins))[:theta_max_all], means_all_theta[:theta_max_all], yerr=std_all_theta[:theta_max_all])
    fig.savefig(path_out_im+stain_clean+"_means_all_theta_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    emb_num=[val for val in list(range(len(files))) for _ in range(1, 2*n_bins-1)]
    
    fig, ax=plt.subplots()
    ax.scatter(list(range(1, n_bins))*len(files)*2, all_means_all_phi.flatten(), marker="o", s=5, alpha=0.5, c=emb_num, cmap="turbo")
    fig.savefig(path_out_im+stain_clean+"_means_all_phi_scatter.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    fig, ax=plt.subplots()
    ax.scatter(list(range(1, n_bins))*len(files)*2, all_means_all_theta.flatten(), marker="o", s=5, alpha=0.5, c=emb_num, cmap="turbo")
    fig.savefig(path_out_im+stain_clean+"_means_all_theta_scatter.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    x_norm = np.linspace(-4, 4,100)
    
    fig, ax=plt.subplots()
    ax.hist(delta_phi_no_nan, bins=25, density=True)
    plt.plot(x_norm,scst.norm.pdf(x_norm,0,1))
    #ax.set_title(shap_delta_phi_pval)
    ax.set_title(kol_delta_phi_pval)
    fig.savefig(path_out_im+stain_clean+"_delta_phi.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    fig, ax=plt.subplots()
    ax.hist(delta_theta_no_nan, bins=25, density=True)
    plt.plot(x_norm,scst.norm.pdf(x_norm,0,1))
    #ax.set_title(shap_delta_theta_pval)
    ax.set_title(kol_delta_theta_pval)
    fig.savefig(path_out_im+stain_clean+"_delta_theta.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    
    x_hist = np.linspace(-4, 4,hist_bins)
    
    fig, ax=plt.subplots()
    ax.errorbar(x_hist, means_delta_phi_hist, yerr=std_delta_phi_hist)
    plt.plot(x_norm,scst.norm.pdf(x_norm,0,1))
    #ax.set_title(shap_delta_phi_pval)
    fig.savefig(path_out_im+stain_clean+"_delta_phi_std.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    fig, ax=plt.subplots()
    ax.errorbar(x_hist, means_delta_theta_hist, yerr=std_delta_theta_hist)
    plt.plot(x_norm,scst.norm.pdf(x_norm,0,1))
    #ax.set_title(shap_delta_theta_pval)
    fig.savefig(path_out_im+stain_clean+"_delta_theta_std.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    