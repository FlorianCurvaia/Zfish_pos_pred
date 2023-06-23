#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:39:00 2023

@author: floriancurvaia
"""


from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

from scipy.stats import zscore

from itertools import product

#import scipy.stats as scst

import numba

import time

#import random as rdm

import matplotlib

matplotlib.rc('xtick', labelsize=5) 
matplotlib.rc('ytick', labelsize=5) 


plt.ioff()

@numba.jit(nopython=True, parallel=True, fastmath=True)
def neigh_med(adj_mat, stain):
    neighbours_median=np.zeros((stain.shape[0]))
    neigh_val=adj_mat*stain
    for i in numba.prange(adj_mat.shape[1]):
        a=neigh_val[i]
        neighbours_median[i]=np.median(a[np.nonzero(a)])
        #neighbours_median[i]=np.mean(a[np.nonzero(a)])
        
    return neighbours_median

@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def chi_g_x(measure, std, mean):
    chisq=(measure-mean)**2/(std**2)
    return(chisq)

@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def P_all_g_given_x(measures, x, all_stds, all_means):
    std=all_stds[x-1]
    mean=all_means[x-1]
    chi_2=chi_g_x(measures, std, mean)
    denominator=math.sqrt(2*math.pi * std**2)
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
def P_x_given_g(n_bins, measures, all_stds, all_means, P_x):
    all_P_g_x=[]
    for x in range(1, n_bins+1):
        local_prob=P_all_g_given_x(measures, x, all_stds, all_means)
        all_P_g_x.append(local_prob*P_x[x-1])
    Z=sum(all_P_g_x)
    
    #all_P_x_g={x:(P_g_x*1/(n_bins-1))/Z for x, P_g_x in all_P_g_x.items()}
    all_P_x_g=[P_g_x/Z for P_g_x in all_P_g_x]
    return(all_P_x_g)

@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def all_posteriors_emb(all_measures, n_bins, all_stds, all_means, P_x):
    posterior_vs_pos=np.zeros((all_measures.shape[0], n_bins))
    for i in numba.prange(all_measures.shape[0]):
        measures=all_measures[i]
        posterior=P_x_given_g(n_bins, measures, all_stds, all_means, P_x)
        posterior_vs_pos[i]=posterior
        
    return posterior_vs_pos



#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/" #"/data/homes/fcurvaia/distances/"
path_in="/data/homes/fcurvaia/distances_new/"
#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
path_out_im="/data/homes/fcurvaia/Images/Posterior/Neighbours/Single_stain/"

path_in_adj="/data/homes/fcurvaia/Spheres_fit/"

#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/Perfect_embryos/"
#path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Perfect_embryos/Posterior/Both_bins/" 

#stain='pSmad1/5_nc_ratio' #, 'pSmad2/3_nc_ratio'
#stain='betaCatenin_nuc'
stain='MapK_nc_ratio'
stain_pred=stain+"_neigh"
c=stain.split("_")
stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])



#wells=["D05", "B06", "C06"] + ["B08", "C08", "D08"] +["D06", "B07", "C07", "D07"]
#wells=["B08", "C08", "D08"] +["D06", "B07", "C07", "D07"]
wells=["D06", "B07", "C07", "D07"] #6 hpf
#wells=["B08", "C08", "D08"] #7 hpf
#wells=["D05", "B06", "C06"] #5_7 hpf
#wells=["C05"] #5_3 hpf
#wells=["B05"] #4_7 hpf
#wells=["B04", "C04", "D04"] #4_3 hpf
#wells=["B03", "C03", "D03"] #3_7 hpf
#wells=["B02", "C02", "D02"] #3_3 hpf

time_emb={"B02":3.3, "C02":3.3, "D02":3.3, "B03":3.7, "C03":3.7, "D03":3.7, "B04":4.3, "C04":4.3, "D04":4.3,
          "B05":4.7, "C05":5.3, "D05":5.7, "B06":5.7, "C06":5.7, "D06":6.0, "B07":6.0, "C07":6.0, "D07":6.0,
          "B08":7.0, "C08":7.0, "D08":7.0}

hpf=time_emb[wells[0]]




n_bins_per_ax=8
n_bins=n_bins_per_ax**2


labels = list(range(1, n_bins+1))
ax_labels=range(1, n_bins_per_ax+1)
theta_bins=np.linspace(0, 1, n_bins_per_ax+1, endpoint=True)
phi_bins_abs=np.linspace(0, math.pi, n_bins_per_ax+1)
phi_labs_abs=360*phi_bins_abs/(2*math.pi)
phi_labs_abs=phi_labs_abs[:-1]


fld=Path(path_in)
files=[]
all_df=[]

to_remove=['D06_px+1741_py-0131', 'D06_px-1055_py-0118', 'B07_px+1257_py-0474', 'C07_px+1929_py-0176', "B07_px-0202_py+0631", "C07_px+0243_py-1998"]
#to_remove=["B07_px-0202_py+0631", "C07_px+0243_py-1998"]

index=list(product(range(1, n_bins_per_ax+1), range(1, n_bins_per_ax+1)))

for w in wells:
    for f in fld.glob(w+"*"+"_w_dist_sph_simp.csv"):
        name=f.name
        emb=name.split("_w")[0]
        if emb in to_remove:
            pass
        else:
            try:
                fn=path_in+emb+"_w_dist_sph_simp.csv"
                feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["theta", "cur_phi"]+[stain])
                files.append(emb)
                feat_filt["emb"]=emb
                adj_mat=np.load(path_in_adj+emb+"_adj_mat.npy")
                np.fill_diagonal(adj_mat, True)
                feat_filt[stain]=zscore(feat_filt[stain])
                stain_neigh=neigh_med(adj_mat, feat_filt[stain].to_numpy())
                feat_filt[stain_pred]=stain_neigh
                y_max=feat_filt[stain].quantile(0.99)
                y_min=feat_filt[stain].quantile(0.025)
                #feat_filt=feat_filt.loc[feat_filt[stain]>y_min]
                #feat_filt=feat_filt.loc[feat_filt[stain]<y_max]
                feat_filt["rel_theta"]=feat_filt.theta/np.max(feat_filt.theta)
                #feat_filt['theta_bin_pos'] = pd.qcut(feat_filt['rel_theta'], n_bins_per_ax, labels=False)+1
                feat_filt['theta_bin_pos'] = pd.to_numeric(pd.cut(x = feat_filt['rel_theta'], bins = theta_bins, labels = ax_labels, include_lowest = False, right=True))
                feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'].abs(), bins = phi_bins_abs, labels = ax_labels, include_lowest = False, right=True))

                feat_filt["both_bins"]=0
                for i in range(len(index)):
                    all_idx=index[i]
                    idx_phi=all_idx[0]
                    idx_theta=all_idx[1]
                    feat_filt.loc[(feat_filt.phi_bin_abs==idx_phi) & (feat_filt.theta_bin_pos==idx_theta), "both_bins"]=i+1
                feat_filt=feat_filt.sort_values(["phi_bin_abs", "theta_bin_pos"])
                feat_filt.reset_index(inplace=True)
                all_df.append(feat_filt)
                #del feat_filt
            except ValueError:
                pass
"""       
stains=["low_dorsal_margin", "high_ventral_margin", "high_margin", "uniform"]
for f in fld.glob("*.csv"):
    name=f.name.split(".")[0]
    emb=name.split("_")[2]
    files.append(emb)
    fn=path_in+"Perfect_embryo_"+emb+".csv"
    feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["theta", "phi"]+stains)
    feat_filt["emb"]=emb
    feat_filt["rel_theta"]=feat_filt.theta/np.max(feat_filt.theta)
    #feat_filt['theta_bin_pos'] = pd.qcut(feat_filt['rel_theta'], n_bins_per_ax, labels=False)+1
    feat_filt['theta_bin_pos'] = pd.to_numeric(pd.cut(x = feat_filt['rel_theta'], bins = theta_bins, labels = ax_labels, include_lowest = False, right=True))
    feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['phi'].abs(), bins = phi_bins_abs, labels = ax_labels, include_lowest = False, right=True))

    feat_filt["both_bins"]=0
    for i in range(len(index)):
        all_idx=index[i]
        idx_phi=all_idx[0]
        idx_theta=all_idx[1]
        feat_filt.loc[(feat_filt.phi_bin_abs==idx_phi) & (feat_filt.theta_bin_pos==idx_theta), "both_bins"]=i+1
    feat_filt=feat_filt.sort_values(["phi_bin_abs", "theta_bin_pos"])
    feat_filt.reset_index(inplace=True)
    all_df.append(feat_filt)
    feat_filt=0
    del feat_filt
"""


feat_filt_all=pd.concat(all_df)

emb_list=np.unique(feat_filt_all.emb)


all_preds=[]

def gen_profiles(emb, feat_filt_all, n_bins, stain):

    profiles_coord={}
    
    mean_profiles_coord=np.zeros((n_bins))

    
    i_mins_coord=[]
    i_maxs_coord=[]
    feat_filt_train=feat_filt_all.loc[(feat_filt_all.emb!=emb)] #(feat_filt_all[ot_coord_bin]==coord_fix) & (feat_filt_all.emb!=emb)
    
    im_diff=np.unique(feat_filt_train.emb)
        
    all_means_all_coord=np.zeros((len(im_diff), n_bins))

    for im, e in zip(im_diff, range(0,len(im_diff))):
        feat_filt=feat_filt_train.loc[feat_filt_train.emb==im]
        
        all_coord_numpy=feat_filt.groupby("both_bins")[stain].mean().to_numpy().copy()
        missing_bins=[]
        for i in range(1, n_bins+1):
            if i not in np.unique(feat_filt.both_bins):
                missing_bins.append(i)
        for i in sorted(missing_bins, reverse=True):
            all_coord_numpy=np.insert(all_coord_numpy, i, 0)
        #all_coord_numpy.resize((n_bins), refcheck=False)
        all_means_all_coord[e]=all_coord_numpy
        
    #randomness=np.random.normal(0,0.05, size=all_means_all_coord.shape[0]*all_means_all_coord.shape[1])
    #randomness=randomness.reshape((all_means_all_coord.shape[0], all_means_all_coord.shape[1]))
    #all_means_all_coord+=randomness

    means_all_coord=np.true_divide(all_means_all_coord.sum(0),(all_means_all_coord!=0).sum(0))

    
    i_min_coord= np.nanmin(means_all_coord[np.nonzero(means_all_coord)])
    i_max_coord=np.nanmax(means_all_coord[np.nonzero(means_all_coord)])
    
    i_mins_coord.append(i_min_coord)
    i_maxs_coord.append(i_max_coord)

    all_means_all_coord=(all_means_all_coord-i_min_coord)/(i_max_coord-i_min_coord)
    
    means_all_coord=(means_all_coord-i_min_coord)/(i_max_coord-i_min_coord)
    
    profiles_coord[stain]=all_means_all_coord
    
    mean_profiles_coord=means_all_coord

        
    
    to_std_coord={std_bin:np.zeros((len(im_diff))) for std_bin in range(1, n_bins+1)}

    
    for pos_bin in range(1, n_bins+1):
        to_std_coord[pos_bin]=profiles_coord[stain].T[pos_bin-1]
    
    std_coord={coo_bin:np.nanstd(mat) for coo_bin, mat in to_std_coord.items()}

    return(mean_profiles_coord, std_coord, i_mins_coord, i_maxs_coord)
    
    

filt=[stain_pred]+["both_bins"]

for embryo in emb_list:
    start_time_0=time.time()
    feat_filt=feat_filt_all.loc[feat_filt_all.emb==embryo][filt]
    to_rm_bins=[]

    feat_filt=feat_filt.sort_values(["both_bins"])
    mean_profiles_coord, std_coord, i_mins_coord, i_maxs_coord = gen_profiles(embryo, feat_filt_all, n_bins, stain_pred)

    P_x_coord=np.histogram(feat_filt.both_bins, bins=n_bins, range=(1, n_bins+1))[0]/len(feat_filt)
    #P_x_coord=np.array((n_bins)*[1/(n_bins)])

    
    for i in range(1, n_bins+1):
        if i not in np.unique(feat_filt.both_bins):
            to_rm_bins.append(i)
            feat_filt.loc[len(feat_filt)]=[0]+[i]
            #feat_filt.loc[len(feat_filt)]=[0,0]+[i]
    feat_filt=feat_filt.sort_values(["both_bins"])
    bins_coord_idx=feat_filt["both_bins"].to_numpy()
    bins_coord_ticks=np.where(bins_coord_idx[:-1] != bins_coord_idx[1:])[0]
    bins_coord_ticks=np.insert(bins_coord_ticks, 0, 0)
            
    measures_coord=feat_filt[stain_pred].to_numpy()
    #measures_coord=feat_filt.groupby("both_bins")[stain].mean().to_numpy().copy()
    
    i_mins_coord=np.array(i_mins_coord)
    i_maxs_coord=np.array(i_maxs_coord)
    measures_coord=(measures_coord-i_mins_coord)/(i_maxs_coord-i_mins_coord)
    posterior_coord=all_posteriors_emb(measures_coord, n_bins, list(std_coord.values()), mean_profiles_coord, P_x_coord)
    print(posterior_coord.shape)
    for i in to_rm_bins:
        posterior_coord[i-1, :]=np.nan
    to_df=posterior_coord.copy()
    #to_df.resize((n_bins,n_bins), refcheck=False)

    preds_t=pd.DataFrame(to_df, columns=labels)
    preds_t["both_bins"]=feat_filt.both_bins.to_numpy()
    all_preds.append(preds_t)
    

    fig, ax=plt.subplots()
    posterior_plot=ax.imshow(posterior_coord, aspect="auto", cmap="inferno")
    #ax.set_yticks(bins_phi_ticks, list(range(1, n_bins))[:len(bins_phi_ticks)])
    ax.set_yticks(bins_coord_ticks, index)  #
    ax.set_xticks(list(range(0, n_bins)), index, rotation=90)
    ax.set_xlabel("Posterior distribution")
    ax.set_ylabel("True bin")
    fig.colorbar(posterior_plot, ax=ax, label="Posterior probability")
    #ax.grid(which='minor', color='w', linestyle='-', linewidth=10)
    ax.set_box_aspect(1)
    fig.savefig(path_out_im+"no_split_posterior_both_bins_"+stain_clean+"_"+embryo+"_"+str(n_bins)+".png", dpi=300) #bbox_inches='tight',
    plt.close()
    
    
    print("1 emb iter --- %s seconds ---\n" % (time.time() - start_time_0))


mean_all_preds=[p.groupby("both_bins")[labels].mean().to_numpy() for p in all_preds]
mean_all_preds=np.nanmean(np.array(mean_all_preds), axis=0)

fig, ax=plt.subplots()
coord_plot=ax.imshow(mean_all_preds, aspect="auto", cmap="inferno")
ax.set_yticks(list(range(0, n_bins)), index)
#ax.set_yticks(list(range(0, n_bins-1)), list(range(1, n_bins)))
ax.set_xticks(list(range(0, n_bins)), index, rotation=90)
ax.set_xlabel("Posterior distribution")
ax.set_ylabel("True bin")
fig.colorbar(coord_plot, ax=ax, label="Mean posterior probability")
#ax.grid(which='minor', color='w', linestyle='-', linewidth=10)
ax.set_box_aspect(1)
fig.savefig(path_out_im+"no_split_posterior_both_bins_mean_"+stain_clean+"_"+str(n_bins)+".png", dpi=300) #bbox_inches='tight',
plt.close()









    
    
    
    
    