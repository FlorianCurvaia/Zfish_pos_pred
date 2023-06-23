#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:12:58 2023

@author: floriancurvaia
"""

import numpy as np

import time

import pandas as pd


import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt

start_time_0=time.time()
n_bins=72
im="B07_px+1257_py-0474" #B04_px+0114_py-1436, B08_px+1076_py-0189, B07_px+1257_py-0474, B03_px-0545_py-1946, B02_px-0280_py+1419
#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/B08_px+1076_py-0189.csv"
fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv" #fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_edge_corr_elli.csv"
#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/features/B02_px-0280_py+1419.csv"
out_ed_samp=pd.read_csv("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_out_ed_samp.csv") #out_ed_samp=pd.read_csv("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_out_ed_samp_elli.csv")
features=pd.read_csv(fn, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "structure", "pSmad2/3_2_Mean", "dist_out", "pSmad1/5_4_Mean", "pSmad1/5_nc_ratio"])

print("Read and load file --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()
feat_filt=features[(features.structure=='cells')]
origin=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/out_sphere_origin_"+im+".npy") #origin=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/out_elli_origin_"+im+".npy")

features[["Centroid_x","Centroid_y", "Centroid_z"]]=features[["Centroid_x","Centroid_y", "Centroid_z"]]-[origin[0],origin[1],origin[2]]
features[["Centroid_x","Centroid_y", "Centroid_z"]]=features[["Centroid_x","Centroid_y", "Centroid_z"]]/[0.65,0.65,1]



def appendSpherical_pd_cm(df): #from https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    xy = df["Centroid_x"]**2+df["Centroid_y"]**2
    df["r"] = np.sqrt(xy + df["Centroid_z"]**2)
    #df["theta"] = pi+np.arctan2(np.sqrt(xy), df["Centroid_z"])
    df["theta"]= np.arccos(df["Centroid_z"]/df["r"])
    #df["phi"] = np.arctan2(df["Centroid_y"], df["Centroid_x"])
    df["phi"] = np.sign(df["Centroid_y"])*np.arccos(df["Centroid_x"]/np.sqrt(xy))
    return df

#features=appendSpherical_pd_cm(features)

print("Generate spherical coordinates --- %s seconds ---\n" % (time.time() - start_time_1))

path_out="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/D_in_microns/"

feat_filt=features[(features.structure=='cells')]
del features


fig0 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color='"pSmad1/5_4_Mean"', opacity=1)
fig0.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig0.write_html(path_out+"pSmad15_4_Mean_"+im+"_"+str(n_bins)+".html") #Modifiy the html file
#fig0.show()



fig8 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="dist_out", opacity=1)
fig8.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig8.write_html(path_out+"dist_out_simp_"+im+"_"+str(n_bins)+".html")



#feat_filt.to_csv("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_sphr_dist.csv", sep=",")











