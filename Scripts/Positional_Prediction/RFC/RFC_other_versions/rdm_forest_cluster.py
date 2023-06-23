#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:11:19 2023

@author: floriancurvaia
"""


import numpy as np

import pandas as pd

from scipy.stats import zscore

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import RepeatedKFold


#B02_px-0280_py+1419, B03_px-0545_py-1946, B04_px+0114_py-1436, B05_px+1522_py-1087, C05_px+0198_py+1683, 
#D05_px-0738_py-1248, B06_px-0493_py+0146, D06_px-1055_py-0118, B07_px+1257_py-0474, B08_px+1076_py-0189
#C04_px-0816_py-1668 INSTEAD OF B04




images=["B02_px-0280_py+1419", "B03_px-0545_py-1946", "C04_px-0816_py-1668", "B05_px+1522_py-1087", "C05_px+0198_py+1683", 
        "D05_px-0738_py-1248", "B06_px-0493_py+0146", "D06_px-1055_py-0118", "B07_px+1257_py-0474", "B08_px+1076_py-0189"]
stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio', 'pSmad2/3_nc_ratio']

to_pred=["theta", "cur_phi", "dist_out"]

path_out_im="/data/homes/fcurvaia/Images/Rdm_forest/"

for im in images:
    
    fn1="/data/homes/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv"
    
    #fn1="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"+im+"_w_dist_sph_simp.csv"
    feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["r_neigh", "NTouchingNeighbors","PhysicalSize", "theta", "dist_out", "cur_phi"]+stains)
    feat_filt.cur_phi=np.abs(feat_filt.cur_phi)
    
    
    train, test= train_test_split(feat_filt[stains+to_pred], test_size=0.3, random_state=42) #[stains+to_pred]
    
    
    
    y_train= pd.concat([train.pop(x) for x in to_pred], axis=1)
    
    x_train=train
    
    
    
    y_test= pd.concat([test.pop(x) for x in to_pred], axis=1)
    
    x_test=test
    
    model = RandomForestRegressor(n_estimators = 200, random_state = 42) #, criterion="absolute_error"
    
    x_all=pd.concat([x_train, x_test], axis=0)
    y_all=pd.concat([y_train, y_test], axis=0)
    
    y_pred = cross_val_predict(model, x_all, y_all, cv=5, n_jobs=5) #model_results.predict(x_test)
    
    
    
    l=len(to_pred)
    
    
    for s in range(l):
        col=to_pred[s]
        
        x_plot=y_all.to_numpy()[:, s]
        y_plot=y_pred[:, s]
        
        f = plt.figure()#
        plt.clf()
        
        axis_min=min(np.min(x_plot), np.min(y_plot))
        axis_max=max(np.max(x_plot), np.max(y_plot))
        plt.xlim(axis_min, axis_max)
        plt.ylim(axis_min, axis_max)
    
        plt.scatter(x=x_plot, y=y_plot, alpha=0.5, s=5)
        plt.plot( [axis_min, axis_max],[axis_min, axis_max] )
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.savefig(path_out_im+im+"_"+col+"_random_forest.png", bbox_inches='tight', dpi=300)
        plt.close()











