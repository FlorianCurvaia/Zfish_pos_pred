#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:45:33 2023

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

import seaborn as sns

import matplotlib.colors as colors

from sklearn.mixture import GaussianMixture

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix


@numba.jit(nopython=True, parallel=True, fastmath=True)
def neigh_med(adj_mat, stain):
    neighbours_median=np.zeros((stain.shape[0]))
    neigh_val=adj_mat*stain
    for i in numba.prange(adj_mat.shape[1]):
        a=neigh_val[i]
        #neighbours_median[i]=np.median(a[np.nonzero(a)])
        neighbours_median[i]=np.mean(a[np.nonzero(a)])
        
    return neighbours_median

"""
@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def chi_g_x(measures, cov_mat, means):
    tot_sum=np.sum(np.transpose((measures-means)[:, np.newaxis])*np.linalg.inv(cov_mat)*(measures-means))
    return(tot_sum)


"""
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
    cov_mat=all_cov_mat[x]
    means=all_means[x,:]
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
    for x in range(0, n_bins):
        local_prob=P_all_g_given_x(measures, x, all_cov_mat, all_means)
        all_P_g_x.append(local_prob*P_x[x])
    Z=sum(all_P_g_x)
    
    #all_P_x_g={x:(P_g_x*1/(n_bins-1))/Z for x, P_g_x in all_P_g_x.items()}
    all_P_x_g=[P_g_x/Z for P_g_x in all_P_g_x]
    return(all_P_x_g)

@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def all_posteriors_emb(all_measures, n_bins, all_cov_mat, all_means, P_x):
    posterior_vs_pos=np.zeros((all_measures.shape[0], n_bins))
    for i in numba.prange(all_measures.shape[0]):
        measures=all_measures[i]
        posterior=P_x_given_g(n_bins, measures, all_cov_mat, all_means, P_x)
        posterior_vs_pos[i]=posterior
        
    return posterior_vs_pos




matplotlib.rc('xtick', labelsize=5) 
matplotlib.rc('ytick', labelsize=5) 


plt.ioff()


path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/" #"/data/homes/fcurvaia/distances/"
#path_in="/data/homes/fcurvaia/distances_new/"

path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/GMM_EM/"
#path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Posterior/Neighbours/Without/"
#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
#path_out_im="/data/homes/fcurvaia/Images/Posterior/Neighbours/"

#path_in_adj="/data/homes/fcurvaia/Spheres_fit/"
path_in_adj="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/"

stains=['MapK_nc_ratio', 'betaCatenin_nuc', 'pSmad1/5_nc_ratio'] #, 'pSmad2/3_nc_ratio' , 'betaCatenin_nuc', 'pSmad1/5_nc_ratio', 'MapK_nc_ratio',
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

"""
filt_ids={}
for stain_pred in stains_pred:
    y_max=feat_filt_all[stain_pred].quantile(0.999)
    y_min=feat_filt_all[stain_pred].quantile(0.025)
    filt_ids[stain_pred]=set(feat_filt_all.loc[(feat_filt_all[stain_pred]<y_max) & (feat_filt_all[stain_pred]>y_min)].index.to_list())
    
to_keep=set.intersection(*list(filt_ids.values()))
feat_filt_all=feat_filt_all.iloc[list(to_keep)]
"""
for stain_pred in stains_pred:
    c=stain_pred.split("_")
    stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
    fig, ax=plt.subplots(figsize=(30,30))
    sns.violinplot(x="both_bins", y=stain_pred, data=feat_filt_all, inner=None, linewidht=0, cut=0, ax=ax)
    fig.savefig("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Posterior/Both_bins/Neighbours/"+stain_clean+"_violin_plot.png", dpi=300)
    plt.close()
    
#plt.hist(feat_filt_all.both_bins, bins=n_bins)
emb_list=np.unique(feat_filt_all.emb)


all_preds=[]

def gen_mean_cov_mat(feat_filt_all, n_bins, stains):
    
    #feat_filt_train=feat_filt_train=feat_filt_all.loc[(feat_filt_all.emb!=emb)]
    
    mean_profiles_coord=feat_filt_all.groupby("both_bins")[stains].mean().to_numpy()
    
    cov_coord=np.zeros((n_bins, len(stains), len(stains)))
    for i in range(1, n_bins+1):
        cov_coord[i-1]=np.cov(feat_filt_all.loc[feat_filt_all.both_bins==i, stains].to_numpy().T, bias=True)


    return(mean_profiles_coord, cov_coord)


def EM_mean_cov_mat(measures, q, n_bins, stains):

    
    mean_profiles_coord=np.zeros((n_bins, len(stains)))
    cov_coord=np.zeros((n_bins, len(stains), len(stains)))
    to_cov_coord=np.zeros((len(stains), measures.shape[0], n_bins))
    for i in range(len(stains)):
        stain=stains[i]
        measures_q=q*measures[stain].to_numpy()[:, np.newaxis]
        means[:,i]=np.mean(measures_q, axis=0)
        to_cov_coord[i]=measures_q
    """
    for i in range(1, n_bins+1):
        
        cov_coord[i-1]=np.cov(to_cov_coord[:,:,i-1].T, bias=True)

    """
    return(mean_profiles_coord, cov_coord)

all_conf_mat=np.zeros((len(emb_list), n_bins, n_bins))


#emb=emb_list[i]
#feat_filt_train=feat_filt_train=feat_filt_all.loc[(feat_filt_all.emb!=emb)]
#feat_filt=feat_filt_all.loc[(feat_filt_all.emb==emb)]
#P_x_coord=np.histogram(feat_filt_all.both_bins, bins=n_bins, range=(1, n_bins+1))[0]/len(feat_filt_all)
P_x_coord=np.histogram(feat_filt_all.both_bins, bins=n_bins, range=(1, n_bins+1))[0]/len(feat_filt_all)
measures=feat_filt_all[stains_pred]
means, cov_mat=gen_mean_cov_mat(feat_filt_all, n_bins, stains_pred)


#train, test= train_test_split(feat_filt_all[stains_pred+["both_bins"]], test_size=0.3, random_state=42)
#Initialization

n_iter=0
while n_iter<3:
    resp=all_posteriors_emb(measures.to_numpy(), n_bins, cov_mat, means, P_x_coord)
    resp=np.nan_to_num(resp)

    #resp=P_x_coord*post/np.sum(P_x_coord*post, axis=1)
    N_k=np.nansum(resp, axis=0)
    for n in range(n_bins):
        cov_n=cov_mat[n]
        for i in range(len(stains_pred)):
            stain_i=stains_pred[i]
            #means[n, i]=np.sum(resp[:,n]*feat_filt_all[stain_i].to_numpy(), axis=0)/N_k[n]
            for j in range(len(stains_pred)):
                stain_j=stains_pred[j]
                cov_n[i, j]=np.sum(resp[:,n]*(feat_filt_all[stain_i].to_numpy()-means[n,i])*(feat_filt_all[stain_j].to_numpy()-means[n,j]), axis=0)/N_k[n]
    
    for n in range(n_bins):
        for i in range(len(stains_pred)):
            stain_i=stains_pred[i]
            means[n, i]=np.sum(resp[:,n]*feat_filt_all[stain_i].to_numpy(), axis=0)/N_k[n]
                       
    #means=feat_filt_copy.groupby("both_bins")[stains].sum().to_numpy()/N_k[:, np.newaxis]   
    #means, cov_mat=EM_mean_cov_mat(measures, resp, n_bins, stains_pred)
    P_x_coord=N_k/len(feat_filt_all)
    n_iter+=1

#resp=all_posteriors_emb(measures.to_numpy(), n_bins, cov_mat, means, P_x_coord)
fig, ax=plt.subplots(figsize=(10,10))
ax.imshow(resp, cmap="inferno", aspect="auto")
ax.set_box_aspect(1)
fig.savefig(path_out_im+"EM_resp.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.close()


MAP=np.zeros_like(resp)
MAP[np.arange(len(resp)), resp.argmax(1)] = 1

conf_mat=confusion_matrix(feat_filt_all["both_bins"], np.where(MAP==1)[1], normalize="true", labels=np.array(range(1, n_bins+1)))
fig, ax=plt.subplots(figsize=(10,10))
MAP_plot=ax.imshow(conf_mat, cmap="inferno")
ax.set_yticks(list(range(0, n_bins)), index)
#ax.set_yticks(list(range(0, n_bins-1)), list(range(1, n_bins)))
ax.set_xticks(list(range(0, n_bins)), index, rotation=90)
fig.colorbar(MAP_plot, ax=ax, label="Mean MAP", shrink=.85)
fig.savefig(path_out_im+"EM_MAP_conf_mat.pdf", dpi=300) #, bbox_inches='tight', pad_inches=0.01
plt.close()

preds_t=pd.DataFrame(resp, columns=labels)
preds_t["both_bins"]=feat_filt_all.both_bins.to_numpy()

mean_posterior=preds_t.groupby("both_bins")[labels].mean().to_numpy()

fig, ax=plt.subplots(figsize=(10,10))
posterior_plot=ax.imshow(mean_posterior, cmap="inferno")
ax.set_yticks(list(range(0, n_bins)), index)
#ax.set_yticks(list(range(0, n_bins-1)), list(range(1, n_bins)))
ax.set_xticks(list(range(0, n_bins)), index, rotation=90)
fig.colorbar(posterior_plot, ax=ax, label="Mean posterior probability", shrink=.85)
fig.savefig(path_out_im+"EM_mean_posterior.pdf", dpi=300) #, bbox_inches='tight', pad_inches=0.01
plt.close()

#E-step
#all_posteriors_emb(feat_filt[stains_pred], n_bins, cov_mat, means, P_x_coord)

#gmm=GaussianMixture(n_components=n_bins, covariance_type="full", max_iter=1000, weights_init=P_x_coord, means_init=means, precisions_init=inv_cov_mat, random_state=42)

"""
#gmm.fit(train[stains_pred])
gmm.fit(feat_filt_train[stains_pred])

#y_pred=gmm.predict(test[stains_pred])
y_pred=gmm.predict(feat_filt[stains_pred])+1

conf_mat=confusion_matrix(feat_filt["both_bins"], y_pred, normalize="true", labels=np.array(range(1, n_bins+1)))
all_conf_mat[i]=conf_mat
fig, ax=plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, ax=ax)
fig.savefig(path_out_im+emb+"_GMM_conf_mat.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.close()

fig, ax=plt.subplots(figsize=(10,10))
sns.heatmap(np.mean(all_conf_mat, axis=0), ax=ax)
ax.set_yticks(list(range(0, n_bins)), index, rotation=90)
#ax.set_yticks(list(range(0, n_bins-1)), list(range(1, n_bins)))
ax.set_xticks(list(range(0, n_bins)), index, rotation=90)
fig.savefig(path_out_im+"GMM_mean_conf_mat.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.close()
"""






