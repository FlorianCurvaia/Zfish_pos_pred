#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:17:07 2023

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

#import matplotlib.colors as colors

#from sklearn.covariance import MinCovDet

from fcit import fcit

from itertools import combinations

from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize

from scipy import stats

matplotlib.rc('xtick', labelsize=5) 
matplotlib.rc('ytick', labelsize=5) 


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
    means=all_means[:, x-1]
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
    for x in range(1, n_bins+1):
        local_prob=P_all_g_given_x(measures, x, all_cov_mat, all_means)
        all_P_g_x.append(local_prob*P_x[x-1])
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

@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def chi_g_x_single(measure, std, mean):
    chisq=(measure-mean)**2/(std**2)
    return(chisq)

"""
@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def P_all_g_given_x_single(measures, std, mean):
    all_probs=np.zeros_like(measures)
    for i in range(len(measures)):
        chi_2=chi_g_x_single(measures[i], std[i], mean[i])
        denominator=math.sqrt(2*math.pi * std[i]**2)
        #print(chi_2)
        numerator=np.exp(-chi_2/2)
        all_probs[i]=numerator/denominator
    prob=np.prod(all_probs)
    return(prob)
"""
@numba.jit(nopython=True, parallel=True, error_model="numpy") #, fastmath=True
def P_all_g_given_x_single(measures, std, mean):
    chi_2=chi_g_x_single(measures, std, mean)
    denominator=np.sqrt(2*math.pi * std**2)
    #print(chi_2)
    numerator=np.exp(-chi_2/2)
    prob=numerator/denominator
    return(prob)

def p_x_given_g_single(measures, means, stds, P_x, n_bins):
    all_p_g_x=np.zeros((measures.shape[0], n_bins))
    for i in range(n_bins):
        p_g_x=np.prod(P_all_g_given_x_single(measures, stds[i, :], means[i, :]), axis=1)
        all_p_g_x[:,i]=p_g_x*P_x[i]
    Z=np.sum(all_p_g_x, axis=1)
    all_p_x_g=all_p_g_x/Z[:, None]
    return(all_p_x_g)


path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/" #"/data/homes/fcurvaia/distances/"
#path_in="/data/homes/fcurvaia/distances_new/"
#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/"

path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Posterior/Neighbours/"
#path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Posterior/Neighbours/Without/"
#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
#path_out_im="/data/homes/fcurvaia/Images/Posterior/Neighbours/"
#path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Posterior/Both_bins/Neighbours/"

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
to_remove_more=['D06_px+1741_py-0131', 'D06_px-1055_py-0118', 'B07_px+1257_py-0474', 'C07_px+1929_py-0176', "D06_px-1055_py-0118"]
to_remove=['C02_px-0797_py+0598',
 'B02_px-2204_py+0773',
 'C02_px+1838_py+0540',
 'B03_px-1274_py+0062',
 'C03_px+1541_py-0312',
 'D03_px-0596_py-1772',
 'B04_px-0190_py+2329',
 'B04_px-2301_py-0402',
 'C04_px+0882_py-1616',
 'D04_px-0512_py-1849',
 'D04_px-1326_py+1186',
 'D04_px+0320_py+0431',
 'D04_px+1315_py+1425',
 'C05_px+0656_py-1139',
 'B05_px-0648_py+0837',
 'B05_px+1541_py+1373',
 'D05_px+1761_py-1171',
 'C07_px+0243_py-1998',
 'B07_px-0202_py+0631',
 'B08_px-0771_py+0185']

#to_remove=to_remove+to_remove_more
#Add B08_px+1160_py-1616 ?

n_neighbours=1 #Smoothening parameter

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
                    feat_filt[stain_pred]=feat_filt[stain]
                    
                for i in range(n_neighbours):
                    for stain_pred in stains_pred:
                        stain_neigh=neigh_med(adj_mat, feat_filt[stain_pred].to_numpy()) #Make dictionnary
                        feat_filt[stain_pred]=stain_neigh
                    
                for stain_pred in stains_pred:
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



check_cond_indep=False

if check_cond_indep==True:
    for c in combinations(range(len(stains_pred)), r=2):
        print(fcit.test(feat_filt_all[stains_pred[c[0]]].to_numpy()[:, None], feat_filt_all[stains_pred[c[1]]].to_numpy()[:, None], feat_filt_all["both_bins"].to_numpy()[:, None]))
"""
for stain_pred in stains_pred:
    feat_filt_all[stain_pred]=zscore(feat_filt_all[stain_pred])
"""
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


def gen_profiles(emb, feat_filt_all, n_bins, stains):

    profiles_coord={}
    
    mean_profiles_coord=np.zeros((len(stains),n_bins))

    
    i_mins_coord=[]
    i_maxs_coord=[]
    feat_filt_train=feat_filt_all.loc[(feat_filt_all.emb!=emb)] #(feat_filt_all[ot_coord_bin]==coord_fix) & (feat_filt_all.emb!=emb)
    
    im_diff=np.unique(feat_filt_train.emb)
    
    for s in range(len(stains)):
        stain=stains[s]
        
        all_means_all_coord=np.zeros((len(im_diff), n_bins))

        for im, e in zip(im_diff, range(0,len(im_diff))):
            feat_filt=feat_filt_train.loc[feat_filt_train.emb==im]
            
            all_coord_numpy=feat_filt.groupby("both_bins")[stain].mean().to_numpy().copy()
            missing_bins=[]
            for i in range(1, n_bins+1):
                if i not in np.unique(feat_filt.both_bins):
                    missing_bins.append(i)
            for i in sorted(missing_bins): #, reverse=True
                all_coord_numpy=np.insert(all_coord_numpy, i, 0) #+missing_bins.index(i)
            #all_coord_numpy.resize((n_bins), refcheck=False)
            all_means_all_coord[e]=all_coord_numpy
            
        
        #randomness=np.random.normal(0,0.05, size=all_means_all_coord.shape[0]*all_means_all_coord.shape[1])
        #randomness=randomness.reshape((all_means_all_coord.shape[0], all_means_all_coord.shape[1]))
        #all_means_all_coord+=randomness

        means_all_coord=np.true_divide(all_means_all_coord.sum(0),(all_means_all_coord!=0).sum(0))

        
        i_min_coord=np.nanmin(means_all_coord[np.nonzero(means_all_coord)])
        i_max_coord=np.nanmax(means_all_coord[np.nonzero(means_all_coord)])
        
        i_mins_coord.append(i_min_coord)
        i_maxs_coord.append(i_max_coord)

        all_means_all_coord=(all_means_all_coord-i_min_coord)/(i_max_coord-i_min_coord)
        
        means_all_coord=(means_all_coord-i_min_coord)/(i_max_coord-i_min_coord)
        
        profiles_coord[stain]=all_means_all_coord
        
        mean_profiles_coord[s]=means_all_coord

        

    to_cov_coord={coo_bin:np.zeros((len(stains), len(im_diff))) for coo_bin in range(1, n_bins+1)}
    #to_cov_coord={}
    
    
    for pos_bin in range(1, n_bins+1):
        for s in range(len(stains)):
            stain=stains[s]
            to_cov_coord[pos_bin][s]=profiles_coord[stain].T[pos_bin-1]
            #to_cov_coord[pos_bin][s]=profiles_coord[stain][pos_bin-1]
            #to_cov_coord[pos_bin]=feat_filt_train.loc[feat_filt_train.both_bins==pos_bin][stains].to_numpy().T
      
    #to_cov_coord={coo_bin:feat_filt_train.loc[feat_filt_train.both_bins==coo_bin, stains].to_numpy() for coo_bin in range(1, n_bins+1)}
    #cov_coord={coo_bin:np.cov(mat, bias=True) for coo_bin, mat in to_cov_coord.items()}
    cov_coord={coo_bin:np.corrcoef(mat) for coo_bin, mat in to_cov_coord.items()}
    #cov_coord={coo_bin:MinCovDet(random_state=42).fit(mat.T).raw_covariance_ for coo_bin, mat in to_cov_coord.items()} #assume_centered=True,

    return(mean_profiles_coord, cov_coord, i_mins_coord, i_maxs_coord)

def covMat(corr_mat, stds):
    cov_mat = np.zeros_like(corr_mat)
    for i in range(corr_mat.shape[0]):
        for j in range(corr_mat.shape[0]):
            # not here that we are just normalizing the covariance matrix
            cov_mat[i][j] =  corr_mat[i][j] * (stds[i] * stds[j])
    return cov_mat


def mse_lhood_single(mean_std, y_measures, x):
    y_lhood=P_all_g_given_x_single(x, mean_std[1], mean_std[0])
    diff=mean_squared_error(y_measures, y_lhood)
    return(diff)

def max_lhood_single(mean_std, y_measures, x):
    y_lhood=P_all_g_given_x_single(x, mean_std[1], mean_std[0])
    diff=mean_squared_error(y_measures, y_lhood)
    return(diff)

def mse_lhood_multi( x, means_cov_mat):
    means=means_cov_mat[1]
    cov_mat=means_cov_mat[1:]
    y_pred=stats.multivariate_normal.pdf(x, means, cov_mat)
    return(y_pred)
    
def gen_mean_cov_mat(emb, feat_filt_all, n_bins, stains):
    
    feat_filt_train=feat_filt_train=feat_filt_all.loc[(feat_filt_all.emb!=emb)]
    
    mean_profiles_coord=feat_filt_train.groupby("both_bins")[stains].mean().to_numpy()
    
    cov_coord=np.zeros((n_bins, len(stains), len(stains)))
    for i in range(1, n_bins+1):
        cov_coord[i-1]=np.cov(feat_filt_train.loc[feat_filt_train.both_bins==i, stains].to_numpy().T, bias=True)


    return(mean_profiles_coord, cov_coord)

count_per_bin=np.histogram(feat_filt_all.both_bins, bins=n_bins, range=(1, n_bins+1))[0]
if check_cond_indep==True:
    all_hist = np.empty([0,4])
    for i in range(1, n_bins+1):
        hist_bins=count_per_bin[i-1]//10
        y_measures=np.apply_along_axis(lambda a: np.histogram(a, bins=hist_bins, density=True)[0], 0, feat_filt_all.loc[feat_filt_all.both_bins==i, stains_pred].to_numpy())
        y_measures=np.hstack((y_measures, np.ones((y_measures.shape[0],1), int)*i))
        all_hist = np.vstack([all_hist ,y_measures])
    
    for c in combinations(range(len(stains_pred)), r=2):
        print(fcit.test(all_hist[:, c[0]][:, None], all_hist[:, c[1]][:, None], all_hist[:, 3].astype(int)[:, None]))

n=len(stains_pred)
mean_profiles_lhood=np.zeros((n_bins, len(stains_pred)))
std_profiles_lhood=np.zeros((n_bins, len(stains_pred)))

for i in range(1, n_bins+1):
    #mean_coord, std_coord, i_mins_coord, i_maxs_coord = gen_mean_std(feat_filt_all, n_bins, stain)
    for j in range(len(stains_pred)):
        stain=stains_pred[j]
        c=stain.split("_")
        stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
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
        mean_profiles_lhood[i-1,j]=mean_lhood
        std_lhood=min_diff_lhood.x[1]
        std_profiles_lhood[i-1,j]=std_lhood


filt=stains_pred+["both_bins"]

MAP=False

all_preds=[]
i_mins_coord=np.min(mean_profiles_lhood, axis=0) #np.array(i_mins_coord)
i_maxs_coord=np.max(mean_profiles_lhood, axis=0)# np.array(i_maxs_coord)
means_to_pred=(mean_profiles_lhood-i_mins_coord)/(i_maxs_coord-i_mins_coord)

for embryo in emb_list:
    start_time_0=time.time()
    feat_filt=feat_filt_all.loc[feat_filt_all.emb==embryo][filt]
    
    feat_filt=feat_filt.sort_values(["both_bins"])
    
    if check_cond_indep==True:
        for c in combinations(range(len(stains_pred)), r=2):
            print(fcit.test(feat_filt[stains_pred[c[0]]].to_numpy()[:, None], feat_filt[stains_pred[c[1]]].to_numpy()[:, None], feat_filt["both_bins"].to_numpy()[:, None]))

    to_rm_bins=[]
    for i in range(1, n_bins+1):
        if i not in np.unique(feat_filt.both_bins):
            to_rm_bins.append(i)
            feat_filt.loc[len(feat_filt)]=[0,0,0]+[i]
            #feat_filt.loc[len(feat_filt)]=[0,0]+[i]
    
    feat_filt=feat_filt.sort_values(["both_bins"])
    measures_coord=feat_filt[stains_pred].to_numpy()
    #measures_coord=feat_filt.groupby("both_bins")[stains].mean().to_numpy().copy()
    
    
    measures_coord=(measures_coord-i_mins_coord)/(i_maxs_coord-i_mins_coord)
    
    
    
    mean_profiles_coord, corr_mat, i_mins_coord_trash, i_maxs_coord_trash = gen_profiles(embryo, feat_filt_all, n_bins, stains_pred)
    
    cov_coord={coo_bin:covMat(mat, stds) for coo_bin, mat, stds in zip(corr_mat.keys(), corr_mat.values(), std_profiles_lhood)}
    
    #mean_profiles_coord, cov_coord = gen_mean_cov_mat(embryo, feat_filt_all, n_bins, stains_pred)

    P_x_coord=np.histogram(feat_filt.both_bins, bins=n_bins, range=(1, n_bins+1))[0]/len(feat_filt)
    #P_x_coord=np.array((n_bins)*[1/(n_bins)])
    
    
    bins_coord_idx=feat_filt["both_bins"].to_numpy()
    bins_coord_ticks=np.where(bins_coord_idx[:-1] != bins_coord_idx[1:])[0]
    bins_coord_ticks=np.insert(bins_coord_ticks, 0, 0)
    
            
    
    posterior_coord=all_posteriors_emb(measures_coord, n_bins, list(cov_coord.values()), means_to_pred.T, P_x_coord) #list(cov_coord.values())
        
    for i in to_rm_bins:
        posterior_coord[i-1, :]=np.nan
    
    to_df=posterior_coord.copy()
    
    if MAP==True:
        to_df=np.zeros_like(posterior_coord)
        to_df[np.arange(len(posterior_coord)), posterior_coord.argmax(1)] = 1
        preds_t=pd.DataFrame(to_df, columns=labels)
        preds_t["both_bins"]=feat_filt.both_bins.to_numpy()
        all_preds.append(preds_t)

        
        #all_preds.append(to_df)
        
    
    
    #to_df.resize((n_bins,n_bins), refcheck=False)
    else:
        preds_t=pd.DataFrame(to_df, columns=labels)
        preds_t["both_bins"]=feat_filt.both_bins.to_numpy()
        all_preds.append(preds_t)
    

    fig, ax=plt.subplots()
    posterior_plot=ax.imshow(to_df, aspect="auto", cmap="inferno")
    #posterior_plot=ax.imshow(to_df, aspect="auto", cmap="inferno")
    #ax.set_yticks(bins_phi_ticks, list(range(1, n_bins))[:len(bins_phi_ticks)])
    if MAP==True:
        ax.set_yticks(list(range(0, n_bins)), index)
    else:
        ax.set_yticks(bins_coord_ticks, index)  #
    ax.set_xticks(list(range(0, n_bins)), index, rotation=90) #
    ax.set_xlabel("Predicted bin")
    ax.set_ylabel("True bin")
    fig.colorbar(posterior_plot, ax=ax, label="Posterior probability")
    ax.set_box_aspect(1)
    fig.savefig(path_out_im+"no_split_posterior_both_bins_"+embryo+"_"+str(n_bins)+".png", dpi=300) #bbox_inches='tight',
    plt.close()
    
    
    print("1 emb iter --- %s seconds ---\n" % (time.time() - start_time_0))

#mean_all_preds=[p.groupby("both_bins")[labels].mean().to_numpy() for p in all_preds]
#mean_all_preds=np.nanmean(np.array(mean_all_preds), axis=0)
mean_all_preds=pd.concat(all_preds, ignore_index=True).groupby("both_bins")[labels].mean().to_numpy()
#mean_all_preds=sum(all_preds)/len(all_preds)

fig, ax=plt.subplots() #figsize=(10,10)
#coord_plot=ax.imshow((mean_all_preds == mean_all_preds.max(axis=0, keepdims=1)).astype(float), aspect="auto", cmap="inferno") #, vmax=0.3
coord_plot=ax.imshow(mean_all_preds, aspect="auto", cmap="inferno") #, vmax=0.3
ax.set_yticks(list(range(0, n_bins)), index)
#ax.set_yticks(list(range(0, n_bins-1)), list(range(1, n_bins)))
ax.set_xticks(list(range(0, n_bins)), index, rotation=90)
ax.set_xlabel("Predicted bin")
ax.set_ylabel("True bin")
fig.colorbar(coord_plot, ax=ax, label="Mean posterior probability")
ax.set_box_aspect(1)
fig.savefig(path_out_im+"no_split_posterior_both_bins_mean_"+str(n_bins)+".png", dpi=300) #bbox_inches='tight',
plt.close()









    
    
    
    
    







    
    
    
    
    