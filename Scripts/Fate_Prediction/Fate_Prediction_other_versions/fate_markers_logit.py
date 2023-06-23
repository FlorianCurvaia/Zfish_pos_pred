#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:33:10 2023

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

#import numba

#import time
plt.rcParams.update({'font.size': 7.5})

fn_fate_markers=Path("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/seurat_files_zfin/Spatial_ReferenceMap.xlsx")

path_in_csv=Path("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/")

path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Fate_markers/Logistic_regression/"

stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio'] #, 'pSmad2/3_nc_ratio'
fate_markers=["gsc", "vent", "noto", "bmp2b", "bmp4", "gata2a", "ta", "lft1"]

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
    df=pd.DataFrame(np.flip(bin_fm.to_numpy(), axis=1), columns=bin_fm.columns)
    df.rename(columns={'1.1': '2'}, inplace=True)
    all_binary_markers[fm_name]=df



all_embs=[]
all_df=[]

to_remove=['D06_px+1741_py-0131', 'D06_px-1055_py-0118', 'B07_px+1257_py-0474', 'C07_px+1929_py-0176', "B07_px-0202_py+0631", "C07_px+0243_py-1998"]
#to_remove=["B07_px-0202_py+0631", "C07_px+0243_py-1998"]


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
                for stain in stains:
                    y_max=feat_filt[stain].quantile(0.99)
                    y_min=feat_filt[stain].quantile(0.025)
                    feat_filt=feat_filt.loc[feat_filt[stain]>y_min]
                    feat_filt=feat_filt.loc[feat_filt[stain]<y_max]
                    feat_filt[stain]=zscore(feat_filt[stain])
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
                        feat_filt.loc[(feat_filt.phi_bin_abs==idx_phi) & (feat_filt.theta_bin_pos==idx_theta), fm]=all_binary_markers[fm].to_numpy()[idx_theta-1, idx_phi-1]
                feat_filt=feat_filt.sort_values(["phi_bin_abs", "theta_bin_pos"])
                feat_filt.reset_index(inplace=True)
                all_df.append(feat_filt)
                del feat_filt
            except ValueError:
                pass

#feat_filt_all=pd.concat(all_df)
feat_filt=pd.concat(all_df)

#for emb in all_embs:
    #feat_filt=feat_filt_all.loc[feat_filt_all.emb==emb]
    
fig, axs=plt.subplots(2, 4, figsize=(30, 30))
fig.subplots_adjust(hspace=0.25)
fig.subplots_adjust(wspace=0.25)

fig1, axs1=plt.subplots(2, 4, figsize=(10, 10))
fig1.subplots_adjust(hspace=0.25)
fig1.subplots_adjust(wspace=0.25)

n_fold=10

for fm, ax, ax1 in zip(fate_markers, axs.flatten(), axs1.flatten()):
    X=feat_filt[stains]
    y=feat_filt[fm]
    logreg=LogisticRegression(penalty=None, random_state=42, solver="lbfgs", class_weight="balanced") #"lbfgs", "newton-cholesky"
    predicted = cross_val_predict(logreg, X, y, cv=n_fold)
    print(fm)
    print(accuracy_score(y, predicted))
    print(classification_report(y, predicted))
    RocCurveDisplay.from_predictions(y, predicted, ax=ax)
    ax.set_title(fm)
    
    conf_mat=confusion_matrix(y, predicted, normalize="true")
    sns.heatmap(conf_mat, cmap="viridis", annot=True, ax=ax1, cbar=False)
    #ax.set_title(im+" "+str(hpf)+" hpf")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")
    ax1.set_title(fm + " 0: "+str(np.sum(y==0))+ " 1: "+str(np.sum(y==1)))
    #ax1.set_yticks()
    
fig.savefig(path_out_im+str(hpf).replace(".", "_")+"_ROC_curves"+"_"+str(n_fold)+"_fold.png", bbox_inches='tight', dpi=300)
plt.close()

fig1.savefig(path_out_im+str(hpf).replace(".", "_")+"_conf_mat"+"_"+str(n_fold)+"_fold.png", bbox_inches='tight', dpi=300)
plt.close()



















