#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:58:57 2023

@author: floriancurvaia
"""


from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

from scipy.stats import zscore

from itertools import product

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import  cross_val_predict #train_test_split

from sklearn.metrics import accuracy_score, classification_report, RocCurveDisplay, confusion_matrix

import seaborn as sns
#from sklearn import metrics, cross_validation

import numba

#import time

@numba.jit(nopython=True, parallel=True, fastmath=True)
def neigh_med(adj_mat, stain):
    neighbours_median=np.zeros((stain.shape[0]))
    neigh_val=adj_mat*stain
    for i in numba.prange(adj_mat.shape[1]):
        a=neigh_val[i]
        neighbours_median[i]=np.median(a[np.nonzero(a)])
        #neighbours_median[i]=np.mean(a[np.nonzero(a)])
        
    return neighbours_median

plt.rcParams.update({'font.size': 10})

#fn_fate_markers=Path("/data/homes/fcurvaia/seurat_files_zfin/Spatial_ReferenceMap.xlsx")
fn_fate_markers=Path("//Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/seurat_files_zfin/Spatial_ReferenceMap.xlsx")

path_in_csv=Path("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/") #"/data/homes/fcurvaia/distances/"
#path_in_csv=Path("/data/homes/fcurvaia/distances_new/")
#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
#path_out_im="/data/homes/fcurvaia/Images/Fate_markers/Logistic_regression/Neighbours/"
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Fate_markers/Correlations/"


#path_in_adj="/data/homes/fcurvaia/Spheres_fit/"
path_in_adj="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/"


stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio'] #, 'pSmad2/3_nc_ratio'
#fate_markers=["gsc", "vent", "noto", "bmp2b", "bmp4", "gata2a", "ta", "lft1"]
fate_markers=["gsc", "noto", "gata2a", "ta", "sebox", "id3"]

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



all_binary_markers=pd.read_excel(fn_fate_markers, sheet_name=None, header=1, index_col=0)

for fm_name, bin_fm in all_binary_markers.items():
    df=pd.DataFrame(np.flip(np.flip(np.flip(bin_fm.to_numpy(), axis=1), axis=0).T, axis=1), columns=bin_fm.columns)
    df.rename(columns={'1.1': '2'}, inplace=True)
    all_binary_markers[fm_name]=df



all_embs=[]
all_df=[]

to_remove=['D06_px+1741_py-0131', 'D06_px-1055_py-0118', 'B07_px+1257_py-0474', 'C07_px+1929_py-0176', "B07_px-0202_py+0631", "C07_px+0243_py-1998"]
#to_remove=["B07_px-0202_py+0631", "C07_px+0243_py-1998"]
stains_pred=[]
for stain in stains:
    stains_pred.append(stain+"_"+"neigh")
    
data_df= pd.read_csv("fate_markers_3.csv", sep=",")
fm_list=list(data_df.columns)
for fm in fm_list:
    arr_=np.reshape(np.array(data_df[fm])[::-1], (8,8)).T[::-1, :].T
    col=arr_.flatten()
    #data_df[fm]=zscore(col)
    data_df[fm]=col

index=list(product(range(1, n_bins_per_ax+1), range(1, n_bins_per_ax+1)))
for w in wells:
    for f in path_in_csv.glob(w+"*"+"_w_dist_sph_simp.csv"):
        name=f.name
        emb=name.split("_w")[0]
        if emb in to_remove:
            pass
        else:
            try:
                fn=path_in_csv / name
                feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["theta", "cur_phi"]+stains)
                all_embs.append(emb)
                feat_filt["emb"]=emb
                #adj_mat=np.load(path_in_adj+emb+"_adj_mat.npy")
                #np.fill_diagonal(adj_mat, True)
                
                for stain in stains:
                    y_max=feat_filt[stain].quantile(0.99)
                    y_min=feat_filt[stain].quantile(0.025)
                    feat_filt=feat_filt.loc[feat_filt[stain]>y_min]
                    feat_filt=feat_filt.loc[feat_filt[stain]<y_max]
                    feat_filt[stain]=zscore(feat_filt[stain])
                """
                for stain, stain_pred in zip(stains, stains_pred):
                    stain_neigh=neigh_med(adj_mat, feat_filt[stain].to_numpy())
                    feat_filt[stain_pred]=stain_neigh
                """
                
                feat_filt["rel_theta"]=feat_filt.theta/np.max(feat_filt.theta)
                #feat_filt['theta_bin_pos'] = pd.qcut(feat_filt['rel_theta'], n_bins_per_ax, labels=False)+1
                feat_filt['theta_bin_pos'] = pd.to_numeric(pd.cut(x = feat_filt['rel_theta'], bins = theta_bins, labels = ax_labels, include_lowest = False, right=True))
                feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'].abs(), bins = phi_bins_abs, labels = ax_labels, include_lowest = False, right=True))

                feat_filt["both_bins"]=0
                feat_filt[fate_markers]=0
                for i in range(len(index)):
                    all_idx=index[i]
                    idx_phi=all_idx[0]
                    idx_theta=all_idx[1]
                    feat_filt.loc[(feat_filt.phi_bin_abs==idx_phi) & (feat_filt.theta_bin_pos==idx_theta), "both_bins"]=i+1
                    for fm in fate_markers:
                        feat_filt.loc[(feat_filt.phi_bin_abs==idx_phi) & (feat_filt.theta_bin_pos==idx_theta), fm]=all_binary_markers[fm].to_numpy()[idx_phi-1, idx_theta-1]
                feat_filt=feat_filt.sort_values(["phi_bin_abs", "theta_bin_pos"])
                feat_filt.reset_index(inplace=True)
                all_df.append(feat_filt)
                del feat_filt
            except ValueError:
                pass

#feat_filt_all=pd.concat(all_df)
feat_filt=pd.concat(all_df)
for stain in stains:
    means_phi_theta=feat_filt.groupby(['phi_bin_abs', 'theta_bin_pos'])[stain].mean().to_numpy()
    data_df[stain]=means_phi_theta
    
fig, axs=plt.subplots(6, 3, figsize=(10, 20), sharex='col', sharey='row')
#print(data_df.corr(method="spearman"))
all_corr=data_df.corr(method="pearson")

for i in range(len(fm_list)):
    fm=fm_list[i]
    for j in range(len(stains)):
        stain=stains[j]
        ax=axs[i,j]
        ax.scatter(data_df[stain], data_df[fm])
        ax.set_title("Pearson r: " + str(np.round(all_corr.loc[fm, stain], 5)))
        ax.set_xlabel(stain)
        ax.set_ylabel("$\it{"+fm.lower()+"}$")
        

fig.savefig(path_out_im+"Correlations_stains_vs_fate_markers_"+str(hpf)+"_hpf.png", dpi=300, bbox_inches='tight') #,
plt.close()
        
#for emb in all_embs:
    #feat_filt=feat_filt_all.loc[feat_filt_all.emb==emb]






















