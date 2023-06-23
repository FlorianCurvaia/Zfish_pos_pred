#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:13:04 2023

@author: floriancurvaia
"""

import numpy as np

import pandas as pd

from scipy.stats import zscore

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

import statsmodels.api as sm

#B02_px-0280_py+1419, B03_px-0545_py-1946, B04_px+0114_py-1436, B05_px+1522_py-1087, C05_px+0198_py+1683, 
#D05_px-0738_py-1248, B06_px-0493_py+0146, D06_px-1055_py-0118, B07_px+1257_py-0474, B08_px+1076_py-0189
#C04_px-0816_py-1668 INSTEAD OF B04


im="C04_px-0816_py-1668"
stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio', 'pSmad2/3_nc_ratio']
fn1="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv"
feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["r_neigh", "NTouchingNeighbors","PhysicalSize", "theta", "dist_out", "cur_phi"]+stains)

to_pred=["theta", "cur_phi", "dist_out"]

def IQR(data, col):
    quant_1= np.quantile(data[col], 0.25)
    quant_3 = np.quantile(data[col], 0.75)
    return(quant_3-quant_1)

def z_score_med(data, col):
    iqr=IQR(data, col)
    data[col]=(data[col]-np.median(data[col]))/iqr
    
feat_filt.cur_phi=np.abs(feat_filt.cur_phi)

"""
for p in to_pred:
    z_score_med(feat_filt,p)

for s in stains:
    z_score_med(feat_filt,s)
"""

"""
for p in to_pred:
    feat_filt[p]=zscore(feat_filt[p], nan_policy="omit")

for s in stains:
    feat_filt[s]=zscore(feat_filt[s], nan_policy="omit")
"""


#feat_filt["dist_out"]=zscore(feat_filt["dist_out"], nan_policy="omit")
#feat_filt.structure.to_numpy()

train, test= train_test_split(feat_filt[stains+to_pred], test_size=0.3, random_state=42)

#train, test= train_test_split(feat_filt, test_size=0.3, random_state=42)

y_train= pd.concat([train.pop(x) for x in to_pred], axis=1)

x_train=sm.add_constant(train)

#x_train = PolynomialFeatures(degree=2, include_bias=False).fit_transform(train)

y_test= pd.concat([test.pop(x) for x in to_pred], axis=1)

x_test=sm.add_constant(test)

#x_test = PolynomialFeatures(degree=2, include_bias=False).fit_transform(test)

l=len(to_pred)
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Lin_reg/"

for s in range(l):
    col=to_pred[s]
    model = sm.GLM(y_train[col], x_train, family=sm.families.Gamma()) #InverseGaussian()
    
    model_results=model.fit()
    
    print(model_results.summary())
    
    
    y_pred = model.predict(model_results.params, x_test)
    
    
    #path_out_im="/data/homes/fcurvaia/Images/UMAP/Common/"
    
    
    
    
    
    x_plot=y_test[col]
    y_plot=y_pred
    
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
    plt.savefig(path_out_im+im+"_"+col+"_glm.png", bbox_inches='tight', dpi=300)
    plt.close()












