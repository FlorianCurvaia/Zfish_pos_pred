#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:48:58 2023

@author: floriancurvaia
"""


import numpy as np

#from math import sqrt, acos

import time

from abbott.h5_files import *

import pandas as pd


from matplotlib import cm, colors
import matplotlib.pyplot as plt

from math import pi

start_time_0=time.time()
#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/B08_px+1076_py-0189.csv"
fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+"B08_px+1076_py-0189"+"_w_dist_edge.csv"
#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/features/B02_px-0280_py+1419.csv"

features=pd.read_csv(fn, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "structure", "pSmad2/3_2_Mean", "borderCentroid", "topCentroid", "dist_out/in_out"])

print("Read and load file --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()
feat_filt=features[(features.structure=='cells')]
origin=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/out_sphere_origin_"+"B08_px+1076_py-0189"+".npy")



features[["Centroid_x","Centroid_y", "Centroid_z"]]=features[["Centroid_x","Centroid_y", "Centroid_z"]]-[origin[0],origin[1],origin[2]]


def appendSpherical_pd(df): #from https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    xy = df["Centroid_x"]**2+df["Centroid_y"]**2
    df["r"] = np.sqrt(xy + df["Centroid_z"]**2)
    #df["theta"] = pi+np.arctan2(np.sqrt(xy), df["Centroid_z"])
    df["theta"]= np.arccos(df["Centroid_z"]/df["r"])
    #df["phi"] = np.arctan2(df["Centroid_y"], df["Centroid_x"])
    df["phi"] = np.sign(df["Centroid_y"])*np.arccos(df["Centroid_x"]/np.sqrt(xy))
    return df

features=appendSpherical_pd(features)

print("Generate spherical coordinates --- %s seconds ---\n" % (time.time() - start_time_1))



feat_filt=features[(features.structure=='cells')]
del features

#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/B08_px+1076_py-0189_w_dist_edge.csv"
#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/features/B02_px-0280_py+1419_sph_coords.csv"

#ratio=pd.read_csv(fn, sep=";", index_col=False, usecols=["dist_out/in_out"]) #



fig3=plt.figure(3)
plt.clf()
ax = fig3.add_subplot(projection='3d')
ax.scatter(feat_filt["Centroid_x"], feat_filt["Centroid_y"], feat_filt["Centroid_z"], c=feat_filt["pSmad2/3_2_Mean"], alpha=0.75)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
cmap = cm.get_cmap("viridis")
norm = colors.Normalize(np.min(feat_filt["pSmad2/3_2_Mean"]), np.max(feat_filt["pSmad2/3_2_Mean"]))
fig3.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)




fig5=plt.figure(5)
plt.clf()
ax = fig5.add_subplot(projection='3d')
ax.scatter(feat_filt["Centroid_x"], feat_filt["Centroid_y"], feat_filt["Centroid_z"], c=feat_filt["theta"], alpha=0.75)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
cmap = cm.get_cmap("viridis")
norm = colors.Normalize(np.min(feat_filt["theta"]), np.max(feat_filt["theta"]))
fig5.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)




fig6=plt.figure(6)
plt.clf()
ax = fig6.add_subplot(projection='3d')
ax.scatter(feat_filt["Centroid_x"], feat_filt["Centroid_y"], feat_filt["Centroid_z"], c=feat_filt["phi"], alpha=0.75)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
cmap = cm.get_cmap("viridis")
norm = colors.Normalize(np.min(feat_filt["phi"]), np.max(feat_filt["phi"]))
fig6.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)


fig9=plt.figure(9)
plt.clf()
ax = fig9.add_subplot(projection='3d')
ax.scatter(feat_filt["Centroid_x"], feat_filt["Centroid_y"], feat_filt["Centroid_z"], c=feat_filt["dist_out/in_out"], alpha=0.75)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
cmap = cm.get_cmap("viridis")
norm = colors.Normalize(np.min(feat_filt["dist_out/in_out"]), np.max(feat_filt["dist_out/in_out"]))
fig9.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
ax.set_xlim3d(-350, 350)
ax.set_ylim3d(-350, 350)
ax.set_zlim3d(-700, 0)


plt.show()













