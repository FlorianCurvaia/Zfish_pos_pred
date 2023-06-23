#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:28:51 2023

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

import random as rdm


plt.ioff()

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


path_in="/data/homes/fcurvaia/distances/"   #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/" #"/data/homes/fcurvaia/distances/" 
path_out_im="/data/homes/fcurvaia/Images/Posterior/Sub_angle/"   #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Posterior/" #"/data/homes/fcurvaia/Images/Pos_inf/"
path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/"
#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Posterior/bins_pred/"

stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio'] #, 'pSmad2/3_nc_ratio'
#wells=["D05", "B06", "C06"] + ["B08", "C08", "D08"] +["D06", "B07", "C07", "D07"]
#wells=["B08", "C08", "D08"] +["D06", "B07", "C07", "D07"]
#wells=["D06", "B07", "C07", "D07"] #6 hpf
wells=["B08", "C08", "D08"] #7 hpf
#wells=["D05", "B06", "C06"] #5_7 hpf
#wells=["C05"] #5_3 hpf
#wells=["B05"] #4_7 hpf
#wells=["B04", "C04", "D04"] #4_3 hpf
#wells=["B03", "C03", "D03"] #3_7 hpf
#wells=["B02", "C02", "D02"] #3_3 hpf


#coord="theta" 
coord="phi"

if coord=="phi":
    coord_bin="phi_bin_abs"
    ot_coord="theta"
    ot_coord_bin="theta_bin_pos"
elif coord=="theta":
    coord_bin="theta_bin_pos"
    ot_coord="phi"
    ot_coord_bin="phi_bin_abs"


n_bins=11

labels = list(range(1, n_bins))
theta_labels=range(1, n_bins)
theta_bins=np.linspace(0, 1, n_bins, endpoint=True)
phi_bins_abs=np.linspace(0, math.pi, n_bins)
phi_labs_abs=360*phi_bins_abs/(2*math.pi)
phi_labs_abs=phi_labs_abs[:-1]


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
            try:
                fn=path_in+emb+"_w_dist_sph_simp.csv"
                feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["theta", "cur_phi"]+stains)
                files.append(emb)
                feat_filt["emb"]=emb
                for stain in stains:
                    y_max=feat_filt[stain].quantile(0.99)
                    y_min=feat_filt[stain].quantile(0.025)
                    feat_filt=feat_filt.loc[feat_filt[stain]>y_min]
                    feat_filt=feat_filt.loc[feat_filt[stain]<y_max]
                    feat_filt[stain]=zscore(feat_filt[stain])
                feat_filt["rel_theta"]=feat_filt.theta/np.max(feat_filt.theta)
                feat_filt['theta_bin_pos'] = pd.to_numeric(pd.cut(x = feat_filt['rel_theta'], bins = theta_bins, labels = theta_labels, include_lowest = False, right=True))

                feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'].abs(), bins = phi_bins_abs, labels = labels, include_lowest = False, right=True))
                all_df.append(feat_filt)
                del feat_filt
            except ValueError:
                pass
        


feat_filt_all=pd.concat(all_df)

"""
to_train=rdm.sample(range(len(all_df)), math.floor(len(all_df)*2/3))
to_test=set(range(len(all_df)))-set(to_train)

feat_filt_all_train=pd.concat([all_df[i] for i in to_train])
feat_filt_all_test=pd.concat([all_df[i] for i in to_test])
"""
emb_list=np.unique(feat_filt_all.emb)


all_preds={}
#for emb in emb_list:
#    all_preds[emb]=[]

def gen_profiles(emb, feat_filt_all, coord_bin, n_bins, stains):

    profiles_phi={}
    
    mean_profiles_phi=np.zeros((len(stains),len(range(n_bins-1))))

    
    i_mins_phi=[]
    i_maxs_phi=[]
    feat_filt_train=feat_filt_all.loc[(feat_filt_all.emb!=emb)] # (feat_filt_all[ot_coord_bin]==theta_fix) & (feat_filt_all.emb!=emb)
    
    im_diff=np.unique(feat_filt_train.emb)
    
    for s in range(len(stains)):
        stain=stains[s]
        
        all_means_all_phi=np.zeros((len(im_diff), n_bins-1))

        for im, e in zip(im_diff, range(0,len(im_diff))):
            feat_filt=feat_filt_train.loc[feat_filt_train.emb==im]
            
            all_phi_numpy=feat_filt.groupby(coord_bin)[stain].mean().to_numpy().copy()
            all_phi_numpy.resize((n_bins-1), refcheck=False)
            all_means_all_phi[e]=all_phi_numpy

        means_all_phi=np.true_divide(all_means_all_phi.sum(0),(all_means_all_phi!=0).sum(0))

        
        i_min_phi= np.nanmin(means_all_phi[np.nonzero(means_all_phi)])
        i_max_phi=np.nanmax(means_all_phi[np.nonzero(means_all_phi)])
        
        i_mins_phi.append(i_min_phi)
        i_maxs_phi.append(i_max_phi)

        all_means_all_phi=(all_means_all_phi-i_min_phi)/(i_max_phi-i_min_phi)
        
        means_all_phi=(means_all_phi-i_min_phi)/(i_max_phi-i_min_phi)
        
        profiles_phi[stain]=all_means_all_phi
        
        mean_profiles_phi[s]=means_all_phi

        
    
    to_cov_phi={phi_bin:np.zeros((len(stains), len(im_diff))) for phi_bin in range(1, n_bins)}

    
    for pos_bin in range(n_bins-1):
        for s in range(len(stains)):
            stain=stains[s]
            to_cov_phi[pos_bin+1][s]=profiles_phi[stain].T[pos_bin]
    
    cov_phi={phi_bin:np.cov(mat, bias=1) for phi_bin, mat in to_cov_phi.items()}

    return(mean_profiles_phi, cov_phi, i_mins_phi, i_maxs_phi)
    
    

    
start_time_0=time.time()
for embryo in emb_list:
    feat_filt=feat_filt_all.loc[feat_filt_all.emb==embryo]
    #feat_filt=feat_filt.loc[feat_filt[ot_coord_bin]==theta_fix]
    
    feat_filt=feat_filt.sort_values([coord_bin])
    mean_profiles_phi, cov_phi, i_mins_phi, i_maxs_phi = gen_profiles(embryo, feat_filt_all, coord_bin, n_bins, stains)
    #mean_profiles_theta, cov_theta, i_mins_theta, i_maxs_theta = gen_profiles(embryo, feat_filt_all, "theta_bin_pos", n_bins, stains)
    
    """
    if coord=="phi":
        P_x_coord=np.histogram(feat_filt.cur_phi.abs(), bins=n_bins-1, range=(0,math.pi))[0]/len(feat_filt)
    elif coord=="theta":
        P_x_coord=np.histogram(feat_filt.rel_theta, bins=n_bins-1, range=(0,1))[0]/len(feat_filt)
    """
    P_x_coord=np.array((n_bins-1)*[1/(n_bins-1)])
    bins_phi_idx=feat_filt[coord_bin].to_numpy()
    bins_phi_ticks=np.where(bins_phi_idx[:-1] != bins_phi_idx[1:])[0]
    bins_phi_ticks=np.insert(bins_phi_ticks, 0, 0)
    
    
    measures_phi=feat_filt[stains].to_numpy()
    measures_phi=feat_filt.groupby(coord_bin)[stains].mean().to_numpy().copy()
    
    i_mins_phi=np.array(i_mins_phi)
    i_maxs_phi=np.array(i_maxs_phi)
    measures_phi=(measures_phi-i_mins_phi)/(i_maxs_phi-i_mins_phi)
    posterior_phi=all_posteriors_emb(measures_phi, n_bins, list(cov_phi.values()), mean_profiles_phi, P_x_coord)
    
    """
    to_df=posterior_phi.copy()
    to_df.resize((n_bins-1,n_bins-1), refcheck=False)
    #posterior[np.isnan(posterior)] = 0
    preds_t=pd.DataFrame(to_df, columns=labels)
    preds_t[coord_bin]=list(range(1, n_bins))
    #preds_t["phi_bin"]=preds_t.index.to_numpy()+1
    preds_t[ot_coord_bin]=theta_fix
    all_preds[embryo].append(preds_t)
    
    """
    all_preds[embryo]=posterior_phi
    fig, ax=plt.subplots()
    ax.imshow(posterior_phi, aspect="auto", cmap="inferno")
    #ax.set_yticks(bins_phi_ticks, list(range(1, n_bins))[:len(bins_phi_ticks)])
    ax.set_yticks(list(range(0, n_bins-1)), list(range(1, n_bins)))
    ax.set_xticks(list(range(0, n_bins-1)), list(range(1, n_bins)))
    ax.set_xlabel("Posterior distribution")
    ax.set_ylabel("True bin")
    ax.set_box_aspect(1)
    fig.savefig(path_out_im+"no_split_posterior_"+coord+"_"+embryo+"_"+str(n_bins)+".png", dpi=300) #bbox_inches='tight',
    plt.close()

    
print("1 bin iter --- %s seconds ---\n" % (time.time() - start_time_0))

    
"""
for embryo in emb_list:
    all_preds[embryo]=pd.concat(all_preds[embryo])
    
for emb  in emb_list:
    feat_filt=all_preds[emb]
    feat_filt=feat_filt.sort_values([coord_bin])
    all_preds[emb]=feat_filt
    bins_phi_idx=feat_filt[coord_bin].to_numpy()
    bins_phi_ticks=np.where(bins_phi_idx[:-1] != bins_phi_idx[1:])[0]
    bins_phi_ticks=np.insert(bins_phi_ticks, 0, 0)
    fig, ax=plt.subplots()
    ax.imshow(feat_filt[labels].to_numpy(), aspect="auto", cmap="inferno")
    ax.set_yticks(bins_phi_ticks, labels[:len(bins_phi_ticks)])
    #ax.set_yticks(list(range(0, n_bins-1)), list(range(1, n_bins)))
    ax.set_xticks(list(range(0, n_bins-1)), labels)
    ax.set_xlabel("Posterior distribution")
    ax.set_ylabel("True bin")
    ax.set_box_aspect(1)
    fig.savefig(path_out_im+"no_split_posterior_"+coord+"_"+emb+"_"+str(n_bins)+".png", dpi=300) #bbox_inches='tight',
    plt.close()
"""

mean_all_preds=[p for p in all_preds.values()] #[labels].to_numpy()
mean_all_preds=np.mean(np.array(mean_all_preds), axis=0)

fig, ax=plt.subplots()
phi=ax.imshow(mean_all_preds, aspect="auto", cmap="inferno")
ax.set_yticks(list(range(0, n_bins-1)), labels)
#ax.set_yticks(list(range(0, n_bins-1)), list(range(1, n_bins)))
ax.set_xticks(list(range(0, n_bins-1)), labels)
ax.set_xlabel("Posterior distribution")
ax.set_ylabel("True bin")
fig.colorbar(phi, ax=ax)
ax.set_box_aspect(1)
fig.savefig(path_out_im+"no_split_posterior_"+coord+"_mean_"+str(n_bins)+".png", dpi=300) #bbox_inches='tight',
plt.close()









    
    
    
    
    