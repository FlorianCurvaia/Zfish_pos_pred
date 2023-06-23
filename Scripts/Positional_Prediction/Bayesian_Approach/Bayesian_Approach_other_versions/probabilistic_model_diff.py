#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:21:45 2023

@author: floriancurvaia
"""




from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

from scipy.stats import zscore

import scipy.stats as scst

import numba

import time

start_time_0=time.time()
path_in="/data/homes/fcurvaia/distances/"   #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/" #"/data/homes/fcurvaia/distances/" 
path_out_im="/data/homes/fcurvaia/Images/Posterior/Diff/"   #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Posterior/" #"/data/homes/fcurvaia/Images/Pos_inf/"

path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Posterior/Diff/"

stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio', 'pSmad2/3_nc_ratio']
#stains=['pSmad1/5_nuc', 'betaCatenin_nuc', 'MapK_nc_ratio', 'pSmad2/3_nc_ratio']
phi_stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc']
theta_stains=['betaCatenin_nuc', 'MapK_nc_ratio']
wells=["D06", "B07", "C07", "D07"]
fld=Path(path_in)
files=[]
all_df=[]
to_remove=['D06_px+1741_py-0131', 'D06_px-1055_py-0118', 'B07_px+1257_py-0474', 'C07_px+1929_py-0176', "B07_px-0202_py+0631", "C07_px+0243_py-1998"]
#to_remove=["B07_px-0202_py+0631", "C07_px+0243_py-1998"]
for w in wells:
    for f in fld.glob(w+"*"+"_w_dist_sph_simp.csv"):
        name=f.name
        emb=name.split("_w")[0]
        if emb in to_remove:
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



#hist_bins=20

n_bins=20
phi_bins=np.linspace(-math.pi, math.pi, n_bins)
labels = range(1, n_bins)
#theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)
theta_bins=np.linspace(0, 1, n_bins, endpoint=True)
phi_bins_abs=np.linspace(0, math.pi, n_bins)
phi_labs_abs=360*phi_bins_abs/(2*math.pi)
phi_labs_abs=phi_labs_abs[:-1]


profiles_phi={}
profiles_theta={}

mean_profiles_phi=np.zeros((len(phi_stains),len(range(n_bins-1))))
mean_profiles_theta=np.zeros((len(theta_stains),len(range(n_bins-1))))

std_profiles_phi={}
std_profiles_theta={}


i_mins_phi=[]
i_maxs_phi=[]
i_mins_theta=[]
i_maxs_theta=[]
all_t_max=[]

for s in range(len(phi_stains)):
    phi_stain=phi_stains[s]
    theta_stain=theta_stains[s]
    c_phi=phi_stain.split("_")
    c_theta=theta_stain.split("_")
    phi_stain_clean="_".join(["".join(c_phi[0].split("/"))]+c_phi[1:])
    theta_stain_clean="_".join(["".join(c_theta[0].split("/"))]+c_theta[1:])
    
    #feat_filt_all[stain]=zscore(feat_filt_all[stain])
    
    #all_means_phi=np.zeros((len(files)*2, n_bins-1))
    #all_means_theta=np.zeros((len(files)*2, n_bins-1))
    all_means_all_phi=np.zeros((len(files), n_bins-1))
    all_means_all_theta=np.zeros((len(files), n_bins-1))
    
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
    for im, e in zip(files, range(0,len(files))):
        fn=path_in+im+"_w_dist_sph_simp.csv"
        #feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["r_neigh", "NTouchingNeighbors","PhysicalSize", "theta", "dist_out","phi", "phi_bin", "theta_bin", "cur_phi", "phi_bin_cur"]+stains)
        feat_filt=feat_filt_all.loc[feat_filt_all.emb==im]
        y_max_phi=feat_filt[phi_stain].quantile(0.975)
        y_min_phi=feat_filt[phi_stain].quantile(0.025)
        feat_filt_phi=feat_filt.loc[feat_filt[phi_stain]>y_min_phi]
        feat_filt_phi=feat_filt_phi.loc[feat_filt_phi[phi_stain]<y_max_phi]
        
        y_max_theta=feat_filt[theta_stain].quantile(0.975)
        y_min_theta=feat_filt[theta_stain].quantile(0.025)
        feat_filt_theta=feat_filt.loc[feat_filt[theta_stain]>y_min_theta]
        feat_filt_theta=feat_filt_theta.loc[feat_filt_theta[theta_stain]<y_max_theta]
        #feat_filt[stain]=zscore(feat_filt[stain])
        
        
        
        #feat_filt.betaCatenin_nuc=zscore(feat_filt.betaCatenin_nuc)
        
        #feat_filt['phi_bin_pos'] = pd.to_numeric(pd.cut(x = feat_filt['phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
        feat_filt_theta['theta_bin_pos'] = pd.to_numeric(pd.cut(x = feat_filt_theta['theta']/np.max(feat_filt_theta['theta']), bins = theta_bins, labels = labels, include_lowest = True, right=True))
        
        feat_filt_phi['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt_phi['cur_phi'].abs(), bins = phi_bins_abs, labels = labels, include_lowest = True, right=False))
        
        #feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
        t_max=np.max(feat_filt_theta.theta_bin_pos)
        all_t_max.append(t_max)

        
        #ax1.errorbar(np.unique(feat_filt_p.phi_bin_abs), feat_filt_p.groupby("phi_bin_abs")[stain].mean(), yerr=feat_filt_p.groupby("phi_bin_abs")[stain].std())
        
        #ax2.errorbar(np.unique(feat_filt_p.theta_bin_pos), feat_filt_p.groupby("theta_bin_pos")[stain].mean(), yerr=feat_filt_p.groupby("theta_bin_pos")[stain].std())
        
        
        all_phi_numpy=feat_filt_phi.groupby("phi_bin_abs")[phi_stain].mean().to_numpy().copy()
        all_phi_numpy.resize((n_bins-1), refcheck=False)
        #all_phi_numpy[all_phi_numpy==0]=np.nan
        #all_means_all_phi[e]=zscore(all_phi_numpy, nan_policy="omit")
        all_means_all_phi[e]=all_phi_numpy
        
        
        all_theta_numpy=feat_filt_theta.groupby("theta_bin_pos")[theta_stain].mean().to_numpy().copy()
        all_theta_numpy.resize((n_bins-1), refcheck=False)
        #all_theta_numpy[all_theta_numpy==0]=np.nan
        #all_means_all_theta[e]=zscore(all_theta_numpy, nan_policy="omit")
        all_means_all_theta[e]=all_theta_numpy

        
        
        

        
    #f1.savefig(path_out_im+stain_clean+"_phi_plot.png", bbox_inches='tight', dpi=300)
    #f2.savefig(path_out_im+stain_clean+"_theta_plot.png", bbox_inches='tight', dpi=300)
    #plt.close()
        
    
    #all_means_phi[all_means_phi == 0] = np.nan
    
    #means_phi=np.nanmean(all_means_phi, axis=0)
    
    #means_all_phi=np.nanmean(all_means_all_phi, axis=0)
    
    #all_means_theta[all_means_theta == 0] = np.nan
    
    #means_theta=np.nanmean(all_means_theta, axis=0)
    
    #means_all_theta=np.nanmean(all_means_all_theta, axis=0)
    
    means_all_phi=np.true_divide(all_means_all_phi.sum(0),(all_means_all_phi!=0).sum(0))
    
    means_all_theta=np.true_divide(all_means_all_theta.sum(0),(all_means_all_theta!=0).sum(0))
    
    phi_max_all=np.argmax(all_means_all_phi==0)
    if phi_max_all==0:
        phi_max_all=n_bins-1
    theta_max_all=np.argmax(all_means_all_theta==0)
    if theta_max_all==0:
        theta_max_all=n_bins-1
    
    i_min_phi= np.nanmin(means_all_phi[np.nonzero(means_all_phi)])
    i_max_phi=np.nanmax(means_all_phi[np.nonzero(means_all_phi)])
    
    i_mins_phi.append(i_min_phi)
    i_maxs_phi.append(i_max_phi)
    
    i_min_theta=np.nanmin(means_all_theta[np.nonzero(means_all_theta)])
    i_max_theta=np.nanmax(means_all_theta[np.nonzero(means_all_theta)])
    
    i_mins_theta.append(i_min_theta)
    i_maxs_theta.append(i_max_theta)
    
    all_means_all_phi=(all_means_all_phi-i_min_phi)/(i_max_phi-i_min_phi)
    
    all_means_all_theta=(all_means_all_theta-i_min_theta)/(i_max_theta-i_min_theta)
    
    means_all_phi=(means_all_phi-i_min_phi)/(i_max_phi-i_min_phi)
    
    means_all_theta=(means_all_theta-i_min_theta)/(i_max_theta-i_min_theta)
    
    #std_phi=np.nanstd(all_means_phi, axis=0)
    
    #std_all_phi=np.nanstd(all_means_all_phi, axis=0)
    
    #std_theta=np.nanstd(all_means_theta, axis=0)
    
    #std_all_theta=np.nanstd(all_means_all_theta, axis=0)
    
    profiles_phi[phi_stain]=all_means_all_phi
    profiles_theta[theta_stain]=all_means_all_theta
    
    mean_profiles_phi[s]=means_all_phi
    mean_profiles_theta[s]=means_all_theta
    
    #std_profiles_phi[stain]=std_all_phi
    #std_profiles_theta[stain]=std_all_theta
    
    
    #feat_filt_all[stain]=feat_filt_all[stain]

to_cov_phi={phi_bin:np.zeros((len(phi_stains), len(files))) for phi_bin in range(1, n_bins)}
to_cov_theta={theta_bin:np.zeros((len(theta_stains), len(files))) for theta_bin in range(1, int(min(all_t_max))+1)}

for pos_bin in range(n_bins-1):
    for s in range(len(phi_stains)):
        phi_stain=phi_stains[s]
        theta_stain=theta_stains[s]
        to_cov_phi[pos_bin+1][s]=profiles_phi[phi_stain].T[pos_bin]
        if pos_bin<min(all_t_max):
            to_cov_theta[pos_bin+1][s]=profiles_theta[theta_stain].T[pos_bin]

cov_phi={phi_bin:np.cov(mat, bias=1) for phi_bin, mat in to_cov_phi.items()}
cov_theta={theta_bin:np.cov(mat, bias=1) for theta_bin, mat in to_cov_theta.items()}


print("Data generation --- %s seconds ---\n" % (time.time() - start_time_0))

#P_x_phi=np.histogram(feat_filt_all.cur_phi.abs(), bins=n_bins, range=(0,math.pi))[0]/len(feat_filt_all)
#P_x_theta=np.histogram(feat_filt_all.theta, bins=n_bins, range=(0,math.pi))[0]/len(feat_filt_all)
    
@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def chi_g_x(measures, cov_mat, means):
    tot_sum=0
    cov_mat_inv=np.linalg.inv(cov_mat)
    for i in range(len(measures)):
        g_i=measures[i]
        g_i_mean=means[i]
        for j in range(len(measures)):
            g_j=measures[j]
            g_j_mean=means[j]
            it_val=(g_i-g_i_mean)*cov_mat_inv[i,j]*(g_j-g_j_mean)
            tot_sum+=it_val
    return(tot_sum)

@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def P_all_g_given_x(measures, x, all_cov_mat, all_means):
    K=len(measures)
    cov_mat=all_cov_mat[x-1]
    means=all_means[:,x-1]
    det=np.linalg.det(cov_mat)
    chi_2=chi_g_x(measures, cov_mat, means)
    denominator=math.sqrt((2*math.pi)**K * det)
    #print(chi_2)
    numerator=math.exp(-chi_2/2)
    prob=numerator/denominator
    return(prob)

def Z_gi(n_bins, measures, all_cov_mat, all_means):
    marginal_prob=0
    for x in range(1, n_bins):
        local_prob=P_all_g_given_x(measures, x, all_cov_mat, all_means)
        marginal_prob+=local_prob
    Z=marginal_prob/(n_bins-1)
    return(Z)
    
@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def P_x_given_g(n_bins, measures, all_cov_mat, all_means, P_x):
    all_P_g_x=[]
    for x in range(1, n_bins):
        local_prob=P_all_g_given_x(measures, x, all_cov_mat, all_means)
        all_P_g_x.append(local_prob*P_x[x-1])
    Z=sum(all_P_g_x)
    
    #all_P_x_g={x:(P_g_x*1/(n_bins-1))/Z for x, P_g_x in all_P_g_x.items()}
    all_P_x_g=[P_g_x/Z for P_g_x in all_P_g_x]
    return(all_P_x_g)

@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def all_posteriors_emb(all_measures, n_bins, all_cov_mat, all_means, P_x):
    posterior_vs_pos=np.zeros((all_measures.shape[0], n_bins-1))
    for i in numba.prange(all_measures.shape[0]):
        measures=all_measures[i]
        posterior=P_x_given_g(n_bins, measures, all_cov_mat, all_means, P_x)
        posterior_vs_pos[i]=posterior
        
    return posterior_vs_pos

i_mins_theta=np.array(i_mins_theta)
i_maxs_theta=np.array(i_maxs_theta)

i_mins_phi=np.array(i_mins_phi)
i_maxs_phi=np.array(i_maxs_phi)

for embryo in files:
    #embryo="D06_px-1055_py-0118" #D06_px-1055_py-0118, B07_px+1257_py-0474
    feat_filt=feat_filt_all.loc[feat_filt_all.emb==embryo]
    feat_filt_phi=feat_filt.copy()
    feat_filt_theta=feat_filt.copy()
    for s in range(len(phi_stains)):
        phi_stain=phi_stains[s]
        theta_stain=theta_stains[s]
        y_max_phi=feat_filt_phi[phi_stain].quantile(0.975)
        y_min_phi=feat_filt_phi[phi_stain].quantile(0.025)
        feat_filt_phi=feat_filt_phi.loc[feat_filt_phi[phi_stain]>y_min_phi]
        feat_filt_phi=feat_filt_phi.loc[feat_filt_phi[phi_stain]<y_max_phi]
        
        y_max_theta=feat_filt_theta[theta_stain].quantile(0.975)
        y_min_theta=feat_filt_theta[theta_stain].quantile(0.025)
        feat_filt_theta=feat_filt_theta.loc[feat_filt_theta[theta_stain]>y_min_theta]
        feat_filt_theta=feat_filt_theta.loc[feat_filt_theta[theta_stain]<y_max_theta]
        #feat_filt[stain]=zscore(feat_filt[stain])
    
    feat_filt_phi['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt_phi['cur_phi'].abs(), bins = phi_bins_abs, labels = labels, include_lowest = True, right=False))
    feat_filt_phi=feat_filt_phi.sort_values("phi_bin_abs")
    P_x_phi=np.histogram(feat_filt_phi.cur_phi.abs(), bins=n_bins-1, range=(0,math.pi))[0]/len(feat_filt_phi)
    P_x_theta=np.histogram(feat_filt_theta.theta/np.max(feat_filt_theta.theta), bins=n_bins-1, range=(0,1))[0]/len(feat_filt_theta)
    
    
    bins_phi_idx=feat_filt_phi.phi_bin_abs.to_numpy()
    bins_phi_ticks=np.where(bins_phi_idx[:-1] != bins_phi_idx[1:])[0]
    bins_phi_ticks=np.insert(bins_phi_ticks, 0, 0)
    
    measures_phi=feat_filt_phi[phi_stains].to_numpy()
    
    
    measures_phi=(measures_phi-i_mins_phi)/(i_maxs_phi-i_mins_phi)
    #measures_phi=zscore(measures_phi, axis=0)
    
    feat_filt_theta['theta_bin_pos'] = pd.to_numeric(pd.cut(x = feat_filt_theta['theta']/np.max(feat_filt_theta['theta']), bins = theta_bins, labels = labels, include_lowest = True, right=True))
    feat_filt_theta=feat_filt_theta.sort_values("theta_bin_pos")
    measures_theta=feat_filt_theta[theta_stains].to_numpy()
    max_theta=feat_filt_theta.theta_bin_pos.max()
    
    bins_theta_idx=feat_filt_theta.theta_bin_pos.to_numpy()
    bins_theta_ticks=np.where(bins_theta_idx[:-1] != bins_theta_idx[1:])[0]
    bins_theta_ticks=np.insert(bins_theta_ticks, 0, 0)
    

    measures_theta=(measures_theta-i_mins_theta)/(i_maxs_theta-i_mins_theta)
    #measures_theta=zscore(measures_theta, axis=0)
    
    
    #idxs=feat_filt_p.loc[feat_filt_p.phi_bin_abs==1].index.tolist()
    #measures=feat_filt_p.loc[idxs[-15]][stains].tolist()
    #measures=np.array(measures)
    
    
    
    start_time_1=time.time()
    posterior_phi=all_posteriors_emb(measures_phi, n_bins, list(cov_phi.values()), mean_profiles_phi, P_x_phi)
    #posterior_theta=all_posteriors_emb(measures_theta, min(all_t_max), list(cov_theta.values()), mean_profiles_theta, P_x_theta)
    posterior_theta=all_posteriors_emb(measures_theta, int(max_theta)+1, list(cov_theta.values()), mean_profiles_theta, P_x_theta)
    prob_time=time.time() - start_time_1
    print("Computing posterior probabilities --- %s seconds ---\n" % (prob_time))
    
    print("Numba posterior probabilities 1 iter --- %s seconds ---\n" % (prob_time/len(feat_filt)))
    
    #posterior[np.isnan(posterior)] = 0
    
    fig, ax=plt.subplots()
    ax.imshow(posterior_phi, aspect="auto", cmap="inferno")
    ax.set_yticks(bins_phi_ticks, list(range(1, n_bins))[:len(bins_phi_ticks)])
    ax.set_xticks(list(range(0, n_bins-1)), list(range(1, n_bins)))
    fig.savefig(path_out_im+"no_split_posterior_phi_"+embryo+"_"+str(n_bins)+".png", bbox_inches='tight', dpi=300)
    plt.close()
        
    
    fig, ax=plt.subplots()
    ax.imshow(posterior_theta, aspect="auto", cmap="inferno")
    ax.set_yticks(bins_theta_ticks, list(range(1, n_bins))[:len(bins_theta_ticks)])
    ax.set_xticks(list(range(0, n_bins-1))[:max_theta], list(range(1, n_bins))[:max_theta])
    fig.savefig(path_out_im+"no_split_posterior_theta_"+embryo+"_"+str(n_bins)+".png", bbox_inches='tight', dpi=300)
    plt.close()
    


    
    
    
    
    
    