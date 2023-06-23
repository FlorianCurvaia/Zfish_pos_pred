#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:28:11 2023

@author: floriancurvaia
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.ensemble import  RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from itertools import product




path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/"
path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Fate_markers/"
sns.set(style="ticks", font_scale=0.5)
n_bins=9

labels = range(1, n_bins)
#theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)
theta_bins=np.linspace(0, 1, n_bins, endpoint=True)
phi_bins_abs=np.linspace(0, math.pi, n_bins)
phi_labs_abs=360*phi_bins_abs/(2*math.pi)
phi_labs_abs=phi_labs_abs[:-1]


phi_labels=[str(x) for x in np.around(phi_labs_abs, 2)]
phi_labels[0]="D "+6*" "+phi_labels[0]
#phi_labels[0]=phi_labels[0][::-1].expandtabs()[::-1]
phi_labels[-1]="V " +phi_labels[-1]
#phi_labels[-1]=phi_labels[-1][::-1].expandtabs()[::-1]

#phi_labels=["" for i in range(len(labels))]
#phi_labels[0]="D"
#phi_labels[-1]="V"

theta_labels=[str(x) for x in labels]
theta_labels[0]=theta_labels[0]+"\nAP"
theta_labels[-1]=theta_labels[-1]+"\nMargin"
#theta_labels[0]="AP\t"+phi_labels[0]

data_df= pd.read_csv("fate_markers_3.csv", sep=",")

fm_insitu={}

for fm in data_df.columns:
    arr_=np.reshape(np.array(data_df[fm])[::-1], (8,8)).T[::-1, :].T
    fm_insitu[fm]=arr_
    
    
    
    
fig, axs=plt.subplots(3,2, figsize=(10, 10))
fig.set_dpi(300)
fig.subplots_adjust(hspace=0.2)
fig.subplots_adjust(wspace=0.15)

for mark, ax_ in zip(list(fm_insitu.keys()),axs.flatten()):
    
    ax=sns.heatmap(fm_insitu[mark], linewidth=0.1, cmap=sns.mpl_palette("viridis", 1000), cbar=True, ax=ax_) #xticklabels=theta_labels, yticklabels=phi_labels,
    #ax.set_xticks(labels, labels)
    #ax.set_yticks(labels, np.around(phi_labs_abs, 2))
    ax.collections[0].colorbar.set_label("$\it{"+mark.lower()+"}$"+" intensity")
    ax_.set_xlabel("theta bin")
    ax_.set_ylabel("phi bin")
    ax_.set_xticks(np.array(labels)-0.5, labels, rotation=0)
    

fig.savefig(path_out_im+"Fate_markers_heatmap.png", bbox_inches='tight', dpi=300)
plt.close(fig)

stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio'] #, 'pSmad2/3_nc_ratio'
im="C05_px+0198_py+1683"
fn=path_in+im+"_w_dist_sph_simp.csv"
feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["theta", "cur_phi"]+stains)
feat_filt["rel_theta"]=feat_filt.theta/feat_filt.theta.max()
feat_filt['theta_bin'] = pd.to_numeric(pd.cut(x = feat_filt['rel_theta'], bins = theta_bins, labels = labels, include_lowest = True, right=True))
feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'].abs(), bins = phi_bins_abs, labels = labels, include_lowest = True, right=False))
for mark, expr in fm_insitu.items():
    feat_filt=feat_filt.sort_values(["phi_bin_abs", "theta_bin"])
    feat_filt[mark]=expr[feat_filt.phi_bin_abs-1, feat_filt.theta_bin-1]
    #feat_filt[mark]=(feat_filt.phi_bin_abs-1, feat_filt.theta_bin-1)
    feat_filt[mark]=feat_filt[mark].factorize()[0]
    #feat_filt[mark]=feat_filt[mark].astype("category")


to_pred=list(fm_insitu.keys())


#model = RandomForestRegressor(n_estimators = 200, random_state = 42)
model = RandomForestClassifier(n_estimators = 200, random_state = 42)

rkf = RepeatedKFold(n_splits=5, n_repeats=10)
feat_filt=feat_filt.sample(frac=1)
x_all=feat_filt[stains]
y_all=feat_filt[to_pred]

y_pred = cross_val_predict(model, x_all, y_all, cv=5, n_jobs=5) #model_results.predict(x_test)


l=len(to_pred)
#path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Lin_reg/"
#path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Reg/"
#path_out_im="/data/homes/fcurvaia/Images/Random_Forest/"


for s in range(l):
    col=to_pred[s]
    
    x_plot=y_all.to_numpy()[:, s]
    y_plot=y_pred[:, s]
    
    conf_mat=confusion_matrix(x_plot, y_plot, normalize="true")
    all_lengths=[]
    for i in range(conf_mat.shape[0]):
        bin_i=conf_mat[i,:]
        left=bin_i[:i].copy()
        left=left[::-1].copy()
        right=bin_i[i:].copy()
        size=max(right.shape[0], left.shape[0])
        right.resize((size))
        left.resize((size), refcheck=False)
        left=np.roll(left, 1)
        cum_sum=np.cumsum(right)+ np.cumsum(left)
        length=np.argmax(cum_sum>2/3)
        all_lengths.append(length)
    
    """
    fig, ax=plt.subplot()
    plt.clf()
    ax.plot(bins_tog[s], all_lengths)
    ax.set_xlabel(col)
    ax.set_ylabel("N bins")
    ax.set_title(im)
    """
    
    
    """
    disp = ConfusionMatrixDisplay.from_predictions(x_plot, y_plot, cmap="turbo", normalize="true")
    disp.plot(ax=ax_conf)
    disp.ax_.set_title(im)
    """
    fig, ax=plt.subplots()
    sns.heatmap(conf_mat, cmap="turbo", annot=False, ax=ax, xticklabels=list(product(range(1, 9), range(1,9))), yticklabels=list(product(range(1, 9), range(1,9))))
    ax.set_title(im+" "+str(col))
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.savefig(path_out_im+str(col)+"_random_forest.png", bbox_inches='tight', dpi=300)
    plt.close()





