#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:03:03 2023

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

import plotly.express as px
#from sklearn import metrics, cross_validation

import numba

#import time

@numba.jit(nopython=True, parallel=True, fastmath=True)
def neigh_med(adj_mat, stain):
    neighbours_median=np.zeros((stain.shape[0]))
    neigh_val=adj_mat*stain
    for i in numba.prange(adj_mat.shape[1]):
        a=neigh_val[i]
        #neighbours_median[i]=np.median(a[np.nonzero(a)])
        neighbours_median[i]=np.mean(a[np.nonzero(a)])
        
    return neighbours_median

plt.rcParams.update({'font.size': 7.5})

#fn_fate_markers=Path("/data/homes/fcurvaia/seurat_files_zfin/Spatial_ReferenceMap.xlsx")
fn_fate_markers=Path("//Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/seurat_files_zfin/Spatial_ReferenceMap.xlsx")

#path_in_csv=Path("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/") #"/data/homes/fcurvaia/distances/"
path_in_csv=Path("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/distances_new/")
#path_in_csv=Path("/data/homes/fcurvaia/distances_new/")
#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
#path_out_im="/data/homes/fcurvaia/Images/Fate_markers/Logistic_regression/Neighbours/"
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Fate_markers/Logistic_regression/Neighbours/"


#path_in_adj="/data/homes/fcurvaia/Spheres_fit/"
path_in_adj="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/"


stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio'] #, 'pSmad2/3_nc_ratio'
#fate_markers=["gsc", "vent", "noto", "bmp2b", "bmp4", "gata2a", "ta", "lft1"]
#fate_markers=["gsc", "vent", "noto", "bmp4", "gata2a", "ta"]
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


"""
all_binary_markers=pd.read_excel(fn_fate_markers, sheet_name=None, header=1, index_col=0)

for fm_name, bin_fm in all_binary_markers.items():
    df=pd.DataFrame(np.flip(np.flip(np.flip(bin_fm.to_numpy(), axis=1), axis=0).T, axis=1), columns=bin_fm.columns)
    df.rename(columns={'1.1': '2'}, inplace=True)
    all_binary_markers[fm_name]=df
"""

data_df= pd.read_csv("fate_markers_3.csv", sep=",")

all_binary_markers={}

for fm_name in data_df.columns:
    arr_=np.reshape(np.array(data_df[fm_name])[::-1], (8,8)).T[::-1, :].T
    all_binary_markers[fm_name.lower()]=pd.DataFrame((arr_>0.3).astype(int))



all_embs=[]
all_df=[]

#to_remove=['D06_px+1741_py-0131', 'D06_px-1055_py-0118', 'B07_px+1257_py-0474', 'C07_px+1929_py-0176', "B07_px-0202_py+0631", "C07_px+0243_py-1998"]
to_remove=["B07_px-0202_py+0631", "C07_px+0243_py-1998"]
stains_pred=[]
for stain in stains:
    stains_pred.append(stain+"_"+"neigh")

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
                feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["theta", "cur_phi", "x_corr", "y_corr", "z_corr"]+stains)
                all_embs.append(emb)
                feat_filt["emb"]=emb
                adj_mat=np.load(path_in_adj+emb+"_adj_mat.npy")
                np.fill_diagonal(adj_mat, True)
                for stain in stains:
                    #y_max=feat_filt[stain].quantile(0.99)
                    #y_min=feat_filt[stain].quantile(0.025)
                    #feat_filt=feat_filt.loc[feat_filt[stain]>y_min]
                    #feat_filt=feat_filt.loc[feat_filt[stain]<y_max]
                    feat_filt[stain]=zscore(feat_filt[stain])
                
                for stain, stain_pred in zip(stains, stains_pred):
                    stain_neigh=neigh_med(adj_mat, feat_filt[stain].to_numpy())
                    feat_filt[stain_pred]=stain_neigh
                    """
                    if stain=="betaCatenin_nuc":
                        feat_filt[stain_pred]=(feat_filt[stain]-stain_neigh)/feat_filt[stain]
                    else:
                        feat_filt[stain_pred]=feat_filt[stain]-stain_neigh
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
emb_list=np.unique(feat_filt.emb)
feat_filt["cur_phi_abs"]=feat_filt.cur_phi.abs()
#for emb in all_embs:
    #feat_filt=feat_filt_all.loc[feat_filt_all.emb==emb]
    
fig, axs=plt.subplots(2, 3, figsize=(12, 7.24))
fig.subplots_adjust(hspace=0.25)
fig.subplots_adjust(wspace=0.25)

fig1, axs1=plt.subplots(2, 3, figsize=(12, 7.24))
fig1.subplots_adjust(hspace=0.25)
fig1.subplots_adjust(wspace=0.25)

fig2, axs2=plt.subplots(2, 3, figsize=(12, 7.24), sharex=True, sharey=True, subplot_kw={'projection': 'polar'})
fig2.subplots_adjust(hspace=0.025)
fig2.subplots_adjust(wspace=0.15)

fig4, axs4=plt.subplots(2, 3, figsize=(12, 7.24), sharex=True, sharey=True, subplot_kw={'projection': 'polar'})
fig4.subplots_adjust(hspace=0.025)
fig4.subplots_adjust(wspace=0.15)

n_fold=5

for fm, ax, ax1, ax2, ax4 in zip(fate_markers, axs.flatten(), axs1.flatten(), axs2.flatten(), axs4.flatten()):
    X=feat_filt[stains_pred]
    y=feat_filt[fm]
    logreg=LogisticRegression(penalty=None, random_state=42, solver="lbfgs", class_weight="balanced") #"lbfgs", "newton-cholesky"
    predicted = cross_val_predict(logreg, X, y, cv=n_fold)
    print(fm)
    print(accuracy_score(y, predicted))
    print(classification_report(y, predicted))
    RocCurveDisplay.from_predictions(y, predicted, ax=ax)
    ax.set_title("$\it{"+fm.lower()+"}$")
    
    conf_mat=confusion_matrix(y, predicted, normalize="true")
    sns.heatmap(conf_mat, cmap="viridis", annot=True, ax=ax1, cbar=False, vmin=0, vmax=1)
    #ax.set_title(im+" "+str(hpf)+" hpf")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")
    ax1.set_title("$\it{"+fm.lower()+"}$" + " 0: "+str(np.sum(y==0))+ " 1: "+str(np.sum(y==1)))
    #ax1.set_yticks()
    feat_filt["diff_lab"]=y-predicted
    means_phi_theta=feat_filt.groupby(['phi_bin_abs', 'theta_bin_pos'])["diff_lab"].mean()
    heatmap_phi_theta=np.full([n_bins_per_ax, n_bins_per_ax], np.nan)
    
    for idx in means_phi_theta.index:
        idx_phi=idx[0]
        idx_theta=idx[1]
        heatmap_phi_theta[idx_phi-1, idx_theta-1]=means_phi_theta.loc[(idx_phi,idx_theta)]
    """
    ax_2=sns.heatmap(np.abs(heatmap_phi_theta), linewidth=0.1, cmap=sns.mpl_palette("viridis", 1000),vmin=0, vmax=1, cbar=True, ax=ax2) #xticklabels=labels, yticklabels=np.around(phi_labs_abs, 2),
    #ax.set_xticks(labels, labels)
    #ax.set_yticks(labels, np.around(phi_labs_abs, 2))
    ax_2.collections[0].colorbar.set_label(" Mean label difference")
    ax2.set_xlabel("theta bin")
    ax2.set_ylabel("phi bin")
    ax2.set_title("$\it{"+fm.lower()+"}$")
    """
    ax_2=ax2.pcolormesh(phi_bins_abs, theta_bins, np.abs(heatmap_phi_theta).T, cmap="viridis", edgecolors='face')
    ax2.pcolormesh(-1*phi_bins_abs, theta_bins, np.abs(heatmap_phi_theta).T, cmap="viridis", edgecolors='face')
    #ax=sns.heatmap(fm_insitu[mark], linewidth=0.1, cmap=sns.mpl_palette("viridis", 1000), cbar=True, ax=ax_) #xticklabels=theta_labels, yticklabels=phi_labels,
    #ax.set_xticks(labels, labels)
    #ax.set_yticks(labels, np.around(phi_labs_abs, 2))
    #ax.collections[0].colorbar.set_label("$\it{"+mark.lower()+"}$"+" intensity")
    ax2.grid(linewidth=0.5)
    ax2.tick_params(pad=0)
    ax2.set_rlabel_position(90)
    ax2.set_yticks(theta_bins, theta_bins)
    ax2.set_xticks(np.pi/180. * np.linspace(180,  -180, 8, endpoint=False))
    ax2.set_thetalim(-np.pi, np.pi)
    ax2.set_title("$\it{"+fm.lower()+"}$")
    fig2.colorbar(ax_2, ax=ax2, label="Mean classification error", pad=0.1, shrink=.75)
    
    
    for emb in emb_list:
        feat_filt_emb=feat_filt.loc[feat_filt.emb==emb].copy()
        feat_filt_emb.loc[(feat_filt_emb.cur_phi_abs<feat_filt_emb.cur_phi_abs.quantile(0.001)) & (feat_filt_emb.diff_lab==0), "diff_lab"]=2
        fig3 = px.scatter_3d(feat_filt_emb, x='x_corr', y='y_corr', z='z_corr', color="diff_lab", opacity=1, color_continuous_scale="viridis")
        fig3.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig3.write_html(path_out_im+"/Embryos_precision/"+fm+"_"+emb+".html")
        fig5 = px.scatter_3d(feat_filt_emb, x='x_corr', y='y_corr', z='z_corr', color=fm, opacity=1, color_continuous_scale="viridis")
        fig5.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig5.write_html(path_out_im+"/FM_location/"+fm+"_"+emb+".html")
    
    ax_4=ax4.pcolormesh(phi_bins_abs, theta_bins, all_binary_markers[fm].T, cmap="viridis", edgecolors='face')
    ax4.pcolormesh(-1*phi_bins_abs, theta_bins, all_binary_markers[fm].T, cmap="viridis", edgecolors='face')
    #ax=sns.heatmap(fm_insitu[mark], linewidth=0.1, cmap=sns.mpl_palette("viridis", 1000), cbar=True, ax=ax_) #xticklabels=theta_labels, yticklabels=phi_labels,
    #ax.set_xticks(labels, labels)
    #ax.set_yticks(labels, np.around(phi_labs_abs, 2))
    #ax.collections[0].colorbar.set_label("$\it{"+mark.lower()+"}$"+" intensity")
    ax4.grid(linewidth=0.5)
    ax4.tick_params(pad=0)
    ax4.set_rlabel_position(90)
    ax4.set_yticks(theta_bins, theta_bins)
    ax4.set_xticks(np.pi/180. * np.linspace(180,  -180, 8, endpoint=False))
    ax4.set_thetalim(-np.pi, np.pi)
    ax4.set_title("$\it{"+fm.lower()+"}$")
    fig4.colorbar(ax_4, ax=ax4, label="Initial binary value", pad=0.1, boundaries=np.linspace(0,1,3, endpoint=True), values=[0,1], ticks=np.linspace(0,1,2), shrink=.75)
    """
    sns.heatmap(all_binary_markers[fm], linewidth=0.1, cmap=sns.mpl_palette("viridis", 1000), cbar=False, ax=ax4) #
    ax4.set_xlabel("theta bin")
    ax4.set_ylabel("phi")
    ax4.set_title("$\it{"+fm.lower()+"}$")
    """
    
fig.savefig(path_out_im+str(hpf).replace(".", "_")+"_ROC_curves"+"_"+str(n_fold)+"_fold.png", bbox_inches='tight', dpi=300)
plt.close()

fig1.savefig(path_out_im+str(hpf).replace(".", "_")+"_conf_mat"+"_"+str(n_fold)+"_fold.png", bbox_inches='tight', dpi=300)
plt.close()

fig2.savefig(path_out_im+str(hpf).replace(".", "_")+"_phi_theta_diff_"+str(n_fold)+"_fold.png", bbox_inches='tight', dpi=300)
plt.close()

fig4.savefig(path_out_im+str(hpf).replace(".", "_")+"_fm_initial_distribution_"+str(n_fold)+"_fold.png", bbox_inches='tight', dpi=300)
plt.close()

















