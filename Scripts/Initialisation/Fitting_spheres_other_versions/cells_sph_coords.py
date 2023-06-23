#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:31:32 2023

@author: floriancurvaia
"""

import numpy as np

#from math import sqrt, acos

import time

import h5py

from joblib import Parallel, delayed

from abbott.h5_files import *

import pandas as pd

from matplotlib.widgets import Slider
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

from math import pi

start_time_0=time.time()
fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/B08_px+1076_py-0189.csv"
#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/features/B02_px-0280_py+1419.csv"

features=pd.read_csv(fn, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "structure", "pSmad2/3_2_Mean", "borderCentroid", "topCentroid" ]) #

print("Read and load file --- %s seconds ---\n" % (time.time() - start_time_0))

"""
fn1='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B08_px+1076_py-0189.h5'
#fn1='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B07_px+1257_py-0474.h5'
with h5py.File(fn1, 'a') as f:
    embryo_dset = h5_select(f, {'stain': "embryo"})[0]
    embryo = to_numpy(embryo_dset)
    zpos = (np.cumsum(embryo, axis=0) * embryo_dset.attrs.get('element_size_um')[0]).astype(np.float32)


fig0=plt.figure(0)
plt.clf()
idx0 = 3
m = plt.imshow(zpos[idx0], vmin=np.min(zpos), vmax=np.max(zpos), cmap="viridis")

axidx = plt.axes([0.25, 0, 0.65, 0.03])
slidx = Slider(axidx, 'z-stack', 0, zpos.shape[0]-1, valinit=idx0, valfmt='%d')

def update(val):
    idx = slidx.val
    m.set_data(zpos[int(idx)])
    fig0.canvas.draw_idle()
slidx.on_changed(update)
"""
"""
start_time_0=time.time()
with h5py.File(fn1) as f:
    for dset in h5_select(f, {'stain': 'cells'}):
        seg_cell = to_numpy(dset)

"""
#sphere_t=np.load("sphere_2.npy")
#sph_points=np.where(sphere_t==True)
start_time_1=time.time()
feat_filt=features[(features.structure=='cells')]
origin=[517.6166872756335806*0.65, 534.9993432631106316*0.65, 513.8579595065110652]
#np.save("Cells_coords", np.array([feat_filt["Centroid_x"], feat_filt["Centroid_y"], feat_filt["Centroid_z"]]))
#np.save("Cells_coords_6", np.array([feat_filt["Centroid_z"], feat_filt["Centroid_y"]/0.65, feat_filt["Centroid_x"]/0.65]))

"""
dist_xy=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/min_dists_x_y.npy")
dist_yx=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/min_dists_y_x.npy")
fig0=plt.figure(0)
plt.clf()
ax = fig0.add_subplot(projection='3d')
ax.scatter(feat_filt["Centroid_x"]/0.65, feat_filt["Centroid_y"]/0.65, feat_filt["Centroid_z"], c=dist_yx, alpha=0.75)
#ax.scatter(sph_points[2], sph_points[1], sph_points[0], alpha=0.25)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim3d(0, 1000)
ax.set_ylim3d(0, 1000)
ax.set_zlim3d(0, 350)
cmap = cm.get_cmap("viridis")
norm = colors.Normalize(np.min(dist_yx), np.max(dist_yx))
fig0.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
ax.view_init(30,220)
"""


features[["Centroid_x","Centroid_y", "Centroid_z"]]=features[["Centroid_x","Centroid_y", "Centroid_z"]]-[origin[0],origin[1],origin[2]]
#features[["Centroid_x","Centroid_y"]]=features[["Centroid_x","Centroid_y"]]*0.65


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

#features.to_csv("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/B08_px+1076_py-0189_sph_coords.csv", sep=";")

#feat_filt=features[(features.structure=='nucleiRaw')]

feat_filt=features[(features.structure=='cells')]

fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/B08_px+1076_py-0189_w_dist_edge.csv"
#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/features/B02_px-0280_py+1419_sph_coords.csv"

ratio=pd.read_csv(fn, sep=";", index_col=False, usecols=["dist_out/in_out"]) #

"""
fig=plt.figure(1)
plt.clf()
ax = fig.add_subplot(projection='3d')
ax.scatter(feat_filt["theta"], feat_filt["phi"], feat_filt["pSmad2/3_2_Mean"], alpha=0.25)

ax.set_xlabel('theta')
ax.set_ylabel('phi')
ax.set_zlabel('pSmad2/3_2_Mean')


fig2=plt.figure(2)
plt.clf()
ax = fig2.add_subplot()
ax.scatter(feat_filt["theta"], feat_filt["phi"], c=feat_filt["pSmad2/3_2_Mean"])
ax.set_xlabel('theta')
ax.set_ylabel('phi')
"""

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
#ax.scatter(origin[0], origin[1], origin[2], color="r", marker="^")



fig4=plt.figure(4)
plt.clf()
ax = fig4.add_subplot(projection='3d')
ax.scatter(feat_filt["Centroid_x"]/0.65, feat_filt["Centroid_y"]/0.65, feat_filt["Centroid_z"], c=feat_filt["pSmad2/3_2_Mean"], alpha=0.75)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
cmap = cm.get_cmap("viridis")
norm = colors.Normalize(np.min(feat_filt["pSmad2/3_2_Mean"]), np.max(feat_filt["pSmad2/3_2_Mean"]))
fig4.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
#ax.scatter(origin[0], origin[1], origin[2], color="r", marker="^")


"""
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
#ax.scatter(origin[0], origin[1], origin[2], color="r", marker="^")



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
#ax.scatter(origin[0], origin[1], origin[2], color="r", marker="^")
"""


fig7=plt.figure(7)
plt.clf()
ax = fig7.add_subplot(projection='3d')
ax.scatter(feat_filt["Centroid_x"], feat_filt["Centroid_y"], feat_filt["Centroid_z"], c=feat_filt["borderCentroid"], alpha=0.75)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
cmap = cm.get_cmap("viridis")
norm = colors.Normalize(np.min(feat_filt["borderCentroid"]), np.max(feat_filt["borderCentroid"]))
fig7.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)



fig8=plt.figure(8)
plt.clf()
ax = fig8.add_subplot(projection='3d')
ax.scatter(feat_filt["Centroid_x"], feat_filt["Centroid_y"], feat_filt["Centroid_z"], c=feat_filt["topCentroid"], alpha=0.75)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
cmap = cm.get_cmap("viridis")
norm = colors.Normalize(np.min(feat_filt["topCentroid"]), np.max(feat_filt["topCentroid"]))
fig8.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
ax.set_xlim3d(-350, 350)
ax.set_ylim3d(-350, 350)
ax.set_zlim3d(-700, 0)

fig9=plt.figure(9)
plt.clf()
ax = fig9.add_subplot(projection='3d')
ax.scatter(feat_filt["Centroid_x"], feat_filt["Centroid_y"], feat_filt["Centroid_z"], c=ratio["dist_out/in_out"], alpha=0.75)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
cmap = cm.get_cmap("viridis")
norm = colors.Normalize(np.min(ratio["dist_out/in_out"]), np.max(ratio["dist_out/in_out"]))
fig8.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
ax.set_xlim3d(-350, 350)
ax.set_ylim3d(-350, 350)
ax.set_zlim3d(-700, 0)


"""
fig7=plt.figure(7)
plt.clf()
plt.hist(feat_filt["theta"])


fig8=plt.figure(8)
plt.clf()
plt.hist(feat_filt["phi"])
"""
plt.show()













