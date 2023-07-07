#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:41:07 2023

@author: floriancurvaia
"""

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

from scipy.stats import zscore, norm

from itertools import product

#import scipy.stats as scst

import numba

import time

#import random as rdm

import matplotlib

import seaborn as sns

import matplotlib.colors as colors

from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize

from scipy import stats

#from scipy.special import rel_entr

#matplotlib.rc('xtick', labelsize=5) 
#matplotlib.rc('ytick', labelsize=5) 


plt.ioff()

@numba.jit(nopython=True, parallel=True, fastmath=True)
def neigh_med(adj_mat, stain):
    neighbours_median=np.zeros((stain.shape[0]))
    neigh_val=adj_mat*stain
    for i in numba.prange(adj_mat.shape[1]):
        a=neigh_val[i]
        #neighbours_median[i]=np.median(a[np.nonzero(a)])
        neighbours_median[i]=np.mean(a[np.nonzero(a)])
        
    return neighbours_median


@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def chi_g_x_single(measure, std, mean):
    chisq=(measure-mean)**2/(std**2)
    return(chisq)

@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def P_all_g_given_x_single(measures, std, mean):
    chi_2=chi_g_x_single(measures, std, mean)
    denominator=math.sqrt(2*math.pi * std**2)
    #print(chi_2)
    numerator=np.exp(-chi_2/2)
    prob=numerator/denominator
    return(prob)




path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/" #"/data/homes/fcurvaia/distances/"
#path_in="/data/homes/fcurvaia/distances_new/"

path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Posterior/Neighbours/"
#path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Posterior/Neighbours/Without/"
#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
#path_out_im="/data/homes/fcurvaia/Images/Posterior/Neighbours/"

#path_in_adj="/data/homes/fcurvaia/Spheres_fit/"
path_in_adj="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/"

stains=['MapK_nc_ratio', 'betaCatenin_nuc', 'pSmad1/5_nc_ratio'] #, 'pSmad2/3_nc_ratio' , 'betaCatenin_nuc', 'pSmad1/5_nc_ratio', 'MapK_nc_ratio',
#stains=['MapK_nuc', 'betaCatenin_nuc', 'pSmad1/5_nuc']
#stains=['MapK_nc_ratio', 'betaCatenin_nuc']
#stains=['MapK_nc_ratio', 'pSmad1/5_nc_ratio']
#stains=['betaCatenin_nuc', 'pSmad1/5_nc_ratio']


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





n_bins_per_ax=8
n_bins=n_bins_per_ax**2

phi_bins=np.linspace(-math.pi, math.pi, n_bins)
labels = list(range(1, n_bins+1))
ax_labels=range(1, n_bins_per_ax+1)
theta_bins=np.linspace(0, 1, n_bins_per_ax+1, endpoint=True)
phi_bins_abs=np.linspace(0, math.pi, n_bins_per_ax+1)
phi_labs_abs=360*phi_bins_abs/(2*math.pi)
phi_labs_abs=phi_labs_abs[:-1]


fld=Path(path_in)
files=[]
all_df=[]
#to_remove=['D06_px+1741_py-0131', 'D06_px-1055_py-0118', 'B07_px+1257_py-0474', 'C07_px+1929_py-0176', "B07_px-0202_py+0631", "C07_px+0243_py-1998", "D06_px-1055_py-0118"]
to_remove=["B07_px-0202_py+0631", "C07_px+0243_py-1998"]


index=list(product(range(1, n_bins_per_ax+1), range(1, n_bins_per_ax+1)))
stains_pred=[]
for stain in stains:
    stains_pred.append(stain+"_"+"neigh")
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
                adj_mat=np.load(path_in_adj+emb+"_adj_mat.npy")
                np.fill_diagonal(adj_mat, True)
                
                for stain in stains:
                    y_max=feat_filt[stain].quantile(0.99)
                    #y_min=feat_filt[stain].quantile(0.025)
                    #feat_filt=feat_filt.loc[feat_filt[stain]>y_min]
                    #feat_filt=feat_filt.loc[feat_filt[stain]<y_max]
                    #feat_filt[stain]=zscore(feat_filt[stain])
                filt_ids={}
                for stain, stain_pred in zip(stains, stains_pred):
                    stain_neigh=neigh_med(adj_mat, feat_filt[stain].to_numpy()) #Make dictionnary
                    feat_filt[stain_pred]=stain_neigh
                    y_max=feat_filt[stain_pred].quantile(1)
                    y_min=feat_filt[stain_pred].quantile(0)
                    filt_ids[stain_pred]=set(feat_filt.loc[(feat_filt[stain_pred]<=y_max) & (feat_filt[stain_pred]>=y_min)].index.to_list())
                
                to_keep=set.intersection(*list(filt_ids.values()))
                feat_filt=feat_filt.iloc[list(to_keep)]
                for stain_pred in stains_pred:
                    feat_filt[stain_pred]=zscore(feat_filt[stain_pred])
                    

                    """
                    if stain=="betaCatenin_nuc":
                        feat_filt[stain_pred]=(feat_filt[stain]-stain_neigh)/feat_filt[stain]
                    else:
                        feat_filt[stain_pred]=feat_filt[stain]-stain_neigh
                    """
                
                    
                feat_filt["rel_theta"]=feat_filt.theta/np.max(feat_filt.theta)
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
                del feat_filt
            except ValueError:
                pass
            

        


feat_filt_all=pd.concat(all_df, ignore_index=True)

norm_bin=[]
for i in range(1, n_bins+1):
    norm_bin.append(zscore(feat_filt_all.loc[feat_filt_all.both_bins==i, stains_pred]))

feat_filt_norm_bin=pd.concat(norm_bin)

fig, axs=plt.subplots(1,3, figsize=(21,7), sharey=True)
for stain, ax in zip(stains_pred, axs.flatten()):
    c=stain.split("_")
    stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
    min_bin=np.min(np.histogram(feat_filt_norm_bin[stain], bins=1000)[1])
    max_bin=np.max(np.histogram(feat_filt_norm_bin[stain], bins=1000)[1])
    hist=np.histogram(feat_filt_norm_bin[stain], bins=1000, density=True)
    #ax.plot(hist[1][:-1], hist[0])
    #ax.plot(hist[1][:-1], hist[0])
    ax.hist(feat_filt_norm_bin[stain], bins=1000, density=True)
    ax.plot(np.linspace(min_bin, max_bin, 600), norm.pdf(np.linspace(min_bin, max_bin, 600), 0, 1))
    if stain[0]=="M":
        ax.set_title("pERK_nc_ratio")
    else:
        ax.set_title(stain.split("_neigh")[0])
    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")
fig.savefig("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Posterior/Both_bins/Neighbours/"+"Hist_fluc_vs_norm.pdf", dpi=300,  bbox_inches='tight', pad_inches=0.01)
plt.close()
    

"""
filt_ids={}
for stain_pred in stains_pred:
    y_max=feat_filt_all[stain_pred].quantile(0.999)
    y_min=feat_filt_all[stain_pred].quantile(0.025)
    filt_ids[stain_pred]=set(feat_filt_all.loc[(feat_filt_all[stain_pred]<y_max) & (feat_filt_all[stain_pred]>y_min)].index.to_list())
    
to_keep=set.intersection(*list(filt_ids.values()))
feat_filt_all=feat_filt_all.iloc[list(to_keep)]
"""
"""
for stain_pred in stains_pred:
    c=stain_pred.split("_")
    stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
    fig, ax=plt.subplots(figsize=(30,30))
    sns.violinplot(x="both_bins", y=stain_pred, data=feat_filt_all, inner=None, linewidht=0, cut=0, ax=ax)
    fig.savefig("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Posterior/Both_bins/Neighbours/"+stain_clean+"_violin_plot.png", dpi=300)
    plt.close()
"""
#plt.hist(feat_filt_all.both_bins, bins=n_bins)
emb_list=np.unique(feat_filt_all.emb)


all_preds=[]

def gen_mean_std(feat_filt_all, n_bins, stains):

    profiles_coord={}
    
    mean_profiles_coord=np.zeros((n_bins))

    
    i_mins_coord=[]
    i_maxs_coord=[]
    
    
    im_diff=np.unique(feat_filt_all.emb)
    

        
    all_means_all_coord=np.zeros((len(im_diff), n_bins))

    for im, e in zip(im_diff, range(0,len(im_diff))):
        feat_filt=feat_filt_all.loc[feat_filt_all.emb==im]
        
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

def diff_lhood(mean, std, y_measures, x):
    y_lhood=P_all_g_given_x_single(x, std, mean)
    diff=(y_measures-y_lhood)**2 #mean_squared_error(y_measures, y_lhood)
    return(diff)

def mse_lhood_single(mean_std, y_measures, x):
    y_lhood=P_all_g_given_x_single(x, mean_std[1], mean_std[0])
    #div=stats.entropy(y_lhood, y_measures)
    diff=mean_squared_error(y_measures, y_lhood)
    return(diff)

def min_loglhood(mean_std, measures):
    y_lhood=np.sum(np.log(P_all_g_given_x_single(measures, mean_std[1], mean_std[0])))
    #std=mean_squared_error(y_measures, y_lhood)
    #LL = np.sum(stats.norm.logpdf(y_measures, y_lhood, std))
    # Calculate the negative log-likelihood
    neg_LL = -1*y_lhood
    return neg_LL 

#hist_bins=100
mean_profiles_coord=np.zeros((n_bins, len(stains_pred)))
count_per_bin=np.histogram(feat_filt_all.both_bins, bins=n_bins, range=(1, n_bins+1))[0]
for j in range(len(stains_pred)):
    stain=stains_pred[j]
    c=stain.split("_")
    stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
    #mean_coord, std_coord, i_mins_coord, i_maxs_coord = gen_mean_std(feat_filt_all, n_bins, stain)
    fig, axs=plt.subplots(n_bins_per_ax, n_bins_per_ax, figsize=(50,50))
    fig1, axs1=plt.subplots(n_bins_per_ax, n_bins_per_ax, figsize=(50,50))
    for i, ax, ax1 in zip(range(1, n_bins+1), axs.flatten(), axs1.flatten()):
        feat_filt_bin=feat_filt_all.loc[feat_filt_all.both_bins==i, [stain, "both_bins"]].copy()
        #feat_filt_bin[stain]=zscore(feat_filt_bin[stain])
        #mean_coord=feat_filt_bin[stain].mean()
        #std_coord=feat_filt_bin[stain].std()
        measures_coord=feat_filt_bin[stain].to_numpy()
        #i_mins_coord=np.array(i_mins_coord)
        #i_maxs_coord=np.array(i_maxs_coord)
        #measures_coord=(measures_coord-i_mins_coord)/(i_maxs_coord-i_mins_coord)
        #measures_coord=zscore(measures_coord)
        mean_coord=np.mean(measures_coord)
        std_coord=np.std(measures_coord)
        hist_bins=count_per_bin[i-1]//10
        y_measures=np.histogram(measures_coord, bins=hist_bins, density=True)[0]
        x_bins=np.histogram(measures_coord, bins=hist_bins)[1]
        x=(x_bins[:-1]+x_bins[1:])/2
        min_diff_lhood=minimize(mse_lhood_single, np.array([mean_coord, std_coord]), args=(y_measures, x), method="L-BFGS-B")
        #min_diff_lhood=minimize(min_loglhood, np.array([mean_coord, std_coord]), args=(measures_coord), method="L-BFGS-B")
        mean_lhood=min_diff_lhood.x[0]
        mean_profiles_coord[i-1,j]=mean_lhood
        std_lhood=min_diff_lhood.x[1]
        #measures_coord=(measures_coord-mean_lhood)/std_lhood
        min_bin=np.min(np.histogram(measures_coord, bins=hist_bins)[1])
        max_bin=np.max(np.histogram(measures_coord, bins=hist_bins)[1])
        ax.hist(measures_coord, bins=hist_bins, density=True)
        ax.plot(np.linspace(min_bin,max_bin,3*hist_bins), P_all_g_given_x_single(np.linspace(min_bin,max_bin,3*hist_bins), std_lhood, mean_lhood))
        ax.set_title("Bin "+str(i))
        ax.set_xlabel("x")
        ax.set_ylabel("P(x)")
        
        #ax1.scatter(x, diff_lhood(mean_lhood, std_lhood, y_measures, x))
        y_prob=P_all_g_given_x_single(x, std_lhood, mean_lhood)
        ax1.scatter(y_measures,y_prob )
        ax1.plot([0,max(max(y_measures),max(y_prob ))],[0,max(max(y_measures),max(y_prob ))] )
        ax1.set_title("Bin "+str(i))
        ax1.set_xlabel("P(x)")
        ax1.set_ylabel("L(x)")
        
    fig.savefig("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Posterior/Both_bins/Neighbours/"+stain_clean+"_hist_max_likelihood_per_bin.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
    fig1.savefig("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Posterior/Both_bins/Neighbours/"+stain_clean+"_diff_likelihood_per_bin.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()








