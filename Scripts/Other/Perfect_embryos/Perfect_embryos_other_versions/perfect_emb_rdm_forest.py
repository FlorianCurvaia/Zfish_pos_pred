#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:13:51 2023

@author: floriancurvaia
"""



import numpy as np

import pandas as pd

from scipy.stats import zscore

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import RepeatedKFold

from math import pi

from sklearn.metrics import confusion_matrix

from sklearn.metrics import ConfusionMatrixDisplay

import time

import seaborn as sns

from pathlib import Path

start_time_0=time.time()


 
def IQR(data, col):
    quant_1= np.quantile(data[col], 0.25)
    quant_3 = np.quantile(data[col], 0.75)
    return(quant_3-quant_1)

def z_score_med(data, col):
    iqr=IQR(data, col)
    data[col]=(data[col]-np.median(data[col]))/iqr
    
    
n_bins_cv=10

stains=["low_dorsal_margin", "high_ventral_margin", "high_margin", "uniform"]

#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/Perfect_embryos/"  #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/" #"/data/homes/fcurvaia/distances/" 
path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/Perfect_embryos/"
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Perfect_embryos/Random_forest/"   #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Posterior/" #"/data/homes/fcurvaia/Images/Pos_inf/"

#path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Perfect_embryos/Random_forest/Shuffle/Labels_only/"

fld=Path(path_in)
files=[]
all_df=[]
#to_remove=['D06_px+1741_py-0131', 'D06_px-1055_py-0118', 'B07_px+1257_py-0474', 'C07_px+1929_py-0176', "B07_px-0202_py+0631", "C07_px+0243_py-1998"]
#to_remove=["B07_px-0202_py+0631", "C07_px+0243_py-1998"]

for f in fld.glob("*.csv"):
    name=f.name.split(".")[0]
    emb=name.split("_")[2]
    files.append(emb)
    fn=path_in+"Perfect_embryo_"+emb+".csv"
    feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["theta", "phi"]+stains)
    feat_filt["emb"]=emb
    all_df.append(feat_filt)
    feat_filt=0
    del feat_filt
    

        
#del files[9]
#del files[4]
#del files[3]
#del files[1]
#files.remove("C07_px+0243_py-1998")
#files.remove('B07_px-0202_py+0631')

feat_filt_all=pd.concat(all_df)

"""
#plt.style.use('dark_background')
f1, axs1 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(60, 30), sharey=True)
fon_size=10
#space_size=38
plt.rcParams.update({'font.size': fon_size}) 
f1.subplots_adjust(hspace=0.125)
f1.subplots_adjust(wspace=0.15)
f1.set_dpi(300)
f2, axs2 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(60, 30), sharey=True)
f2.subplots_adjust(hspace=0.125)
f2.subplots_adjust(wspace=0.125)
f2.set_dpi(300)
f3, axs3 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(60, 30))
f3.subplots_adjust(hspace=0.125)
f3.subplots_adjust(wspace=0.125)
f3.set_dpi(300)
f4, axs4 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(60, 30))
f4.subplots_adjust(hspace=0.125)
f4.subplots_adjust(wspace=0.125)
f4.set_dpi(300)
"""

fon_size=10
#space_size=38
plt.rcParams.update({'font.size': fon_size}) 

#, ax1, ax2, ax3, ax4
for im in files: #, axs1.flatten(), axs2.flatten(), axs3.flatten(), axs4.flatten()):
    #stains=['betaCatenin_nuc', 'pMyosin_nc_ratio', 'MapK_nc_ratio', 'pSmad2/3_nc_ratio', 'pSmad1/5_nc_ratio', 'Pol2-S5P_nc_ratio', 'mTOR-pS6-H3K27AC_nc_ratio']
    #fn1="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv"
    #fn1="/data/homes/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv"
    #fn1="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"+im+"_w_dist_sph_simp.csv"
    #feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["r_neigh", "NTouchingNeighbors","PhysicalSize", "theta", "dist_out", "cur_phi", "phi_bin_cur", "theta_bin"]+stains)
    feat_filt=feat_filt_all.loc[feat_filt_all.emb==im]

    #well=im.split("_")[0]
    #hpf=time_emb[well]
    #to_pred=["theta", "cur_phi", "dist_out"]
    
    #to_pred=["theta_bin", "phi_bin_cur"]
   
        
    
    phi_bins=np.linspace(-pi, pi, n_bins_cv)
    labels = range(1, n_bins_cv)
    theta_bins=np.linspace(0, pi/2, n_bins_cv, endpoint=True)
    
    feat_filt['phi_bin_cv'] = pd.to_numeric(pd.cut(x = feat_filt['phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
    feat_filt['theta_bin_cv'] = pd.to_numeric(pd.cut(x = feat_filt['theta'], bins = theta_bins, labels = labels, include_lowest = True, right=False))
    
    phi_bins_abs=np.linspace(0, pi, n_bins_cv)
    feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['phi'].abs(), bins = phi_bins_abs, labels = labels, include_lowest = True, right=False))
    
    #feat_filt[['theta_bin_cv', 'phi_bin_abs']]=feat_filt[['theta_bin_cv', 'phi_bin_abs']].sample(frac=1).values
    
    t_max=np.max(feat_filt.theta_bin_cv)
    theta_labs=360*theta_bins/(2*pi)
    theta_labs=theta_labs[:-1]
    phi_labs_abs=360*phi_bins_abs/(2*pi)
    phi_labs_abs=phi_labs_abs[:-1]
    
    phi_labs=360*phi_bins/(2*pi)
    phi_labs=phi_labs[:-1]
        
    
    
    bins_tog=[theta_labs, phi_labs_abs]
    
    to_pred=["theta_bin_cv", "phi_bin_abs"]
    
    feat_filt.phi=np.abs(feat_filt.phi)
    
    
    train, test= train_test_split(feat_filt[stains+to_pred], test_size=0.3, random_state=42) #[stains+to_pred]
    
    #train, test= train_test_split(feat_filt, test_size=0.3, random_state=42)
    
    y_train= pd.concat([train.pop(x) for x in to_pred], axis=1)
    
    x_train=train
    
    #x_train = PolynomialFeatures(degree=2, include_bias=False).fit_transform(train)
    
    y_test= pd.concat([test.pop(x) for x in to_pred], axis=1)
    
    x_test=test
    
    
    model = RandomForestClassifier(n_estimators = 200, random_state = 42)
    
    x_all=pd.concat([x_train, x_test], axis=0)
    y_all=pd.concat([y_train, y_test], axis=0)
    
    #model_results=model.fit(x_train, y_train)
    
    rkf = RepeatedKFold(n_splits=5, n_repeats=10)
    y_pred = cross_val_predict(model, x_all, y_all, cv=5, n_jobs=5) #model_results.predict(x_test)
    
    #print(model.score(x_test, y_test))
    #path_out_im="/data/homes/fcurvaia/Images/UMAP/Common/"
    
    
    #x_test = PolynomialFeatures(degree=2, include_bias=False).fit_transform(test)
    
    l=len(to_pred)
    #path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Lin_reg/"
    #path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Reg/"
    #path_out_im="/data/homes/fcurvaia/Images/Random_Forest/"
    
    #axs_sub=[ax1, ax2]
    #axs_conf=[ax3, ax4]
    #, sub_ax, ax_conf 
    for s in range(l): #, axs_sub, axs_conf
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
        
        fig, ax=plt.subplots()
        ax.plot(bins_tog[s], all_lengths)
        ax.set_xlabel(col)
        ax.set_ylabel("N bins")
        #ax.set_title(im)
        fig.savefig(path_out_im+im+"_"+col+"_plot_random_forest.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        
        """
        disp = ConfusionMatrixDisplay.from_predictions(x_plot, y_plot, cmap="turbo", normalize="true")
        disp.plot(ax=ax_conf)
        disp.ax_.set_title(im)
        """
        fig, ax=plt.subplots()
        sns.heatmap(conf_mat, cmap="turbo", annot=True, ax=ax)
        #ax.set_title(im+" "+str(hpf)+" hpf")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        fig.savefig(path_out_im+im+"_"+col+"_donf_mat_random_forest.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        
print("Running time --- %s seconds ---\n" % (time.time() - start_time_0))










