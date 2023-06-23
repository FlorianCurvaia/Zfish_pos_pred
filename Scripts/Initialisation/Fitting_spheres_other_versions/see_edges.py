#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:27:07 2023

@author: floriancurvaia
"""

import numpy as np
import time
import scipy.ndimage as ndimage
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import pandas as pd

path_in='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/'
im="B04_px+0114_py-1436"
edges=np.load(path_in+"edges_"+im+".npy")


fn1="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv"
features=pd.read_csv(fn1, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "structure", "pSmad2/3_2_Mean", "theta","phi", "phi_bin", "theta_bin", "dist_out"])
feat_filt=features[(features.structure=='cells')]
del features
#feat_filt[["Centroid_x","Centroid_y", "Centroid_z"]]=feat_filt[["Centroid_x","Centroid_y", "Centroid_z"]]/[0.65, 0.65, 1]
#feat_filt[["Centroid_x","Centroid_y", "Centroid_z"]]=feat_filt[["Centroid_x","Centroid_y", "Centroid_z"]]+[origin[0],origin[1],origin[2]]
mask=np.copy(edges)
edges[~mask] = 0

#Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays

struct = ndimage.generate_binary_structure(3, 3)
erode = ndimage.binary_erosion(mask, struct)
out_ed = mask ^ erode


#out_edges=np.load("Arrays/out_edges_"+im+".npy")
#out_elli=np.load(path_in+"out_elli_"+im+".npy")
in_sphere=np.load(path_in+"mid_sphere_"+im+".npy")
out_sphere=np.load(path_in+"out_sphere_"+im+".npy")

feat_filt[["Centroid_x","Centroid_y", "Centroid_z"]]=feat_filt[["Centroid_x","Centroid_y", "Centroid_z"]]/[0.65,0.65,1]
cells_coords=feat_filt[["Centroid_x","Centroid_y", "Centroid_z"]].to_numpy().T.astype(int)
start_time_5=time.time()
points=np.zeros((out_sphere.shape))
points_1=np.zeros((out_sphere.shape))
#points=np.zeros((307,1000,1000))
points_1[cells_coords[2], cells_coords[1], cells_coords[0]]=True
points[cells_coords[2], cells_coords[1], cells_coords[0]]=feat_filt.dist_out.to_numpy()
"""
sum_2D_points=[]
sum_2D_edges=[]
for i in range(len(points)):
    sum_2D_points.append(np.sum(points[i]))
    sum_2D_edges.append(np.sum(edges[i]))
fig3=plt.figure(3)
plt.clf()
plt.plot(range(len(points)), sum_2D_points)
"""
r = 10

xx, yy = np.mgrid[-r:r+1, -r:r+1]
circle = xx**2 + yy**2 <= r**2

#Source: https://stackoverflow.com/questions/71875073/draw-a-circle-in-a-numpy-array-given-index-and-radius-without-external-libraries
for point, point1 in zip(points, points_1):
    condition = np.where(point1==True) # np.where(arr > .5) to benchmark 60k circles
    for y,x in zip(*condition):
        # valid indices of the array
        i = slice(max(x-r,0), min(x+r+1, point.shape[1]))
        j = slice(max(y-r,0), min(y+r+1, point.shape[0]))
    
        # visible slice of the circle
        ci = slice(abs(min(x-r, 0)), circle.shape[1] - abs(min(point.shape[1]-(x+r+1), 0)))
        cj = slice(abs(min(y-r, 0)), circle.shape[0] - abs(min(point.shape[0]-(y+r+1), 0)))
        c_i_j=circle[cj, ci]>0
        
        point1[j, i] += c_i_j.astype("float64")*np.max(point[j,i])
    #point1=point1.T
print("get circle on chosen points--- %s seconds ---\n" % (time.time() - start_time_5))





fig1=plt.figure(1)
plt.clf()
idx0 = 3
#m = plt.imshow(out_elli[idx0], vmin=0, vmax=1)
k = plt.imshow(points_1[idx0], vmin=0, vmax=np.max(feat_filt.dist_out), alpha=1, cmap="magma")
n = plt.imshow(out_ed[idx0], vmin=0, vmax=1, alpha=0.5)
p = plt.imshow(in_sphere[idx0], vmin=0, vmax=1, alpha=0.25, cmap="magma")

 #vmin=np.min(points), vmax=np.max(points), alpha=0.75)
o = plt.imshow(out_sphere[idx0], vmin=0, vmax=1, alpha=0.25)

axidx = plt.axes([0.25, 0, 0.65, 0.03])
slidx = Slider(axidx, 'z-stack', 0, out_ed.shape[0]-1, valinit=idx0, valfmt='%d')

def update(val):
    idx = slidx.val
    #m.set_data(out_elli[int(idx)])
    n.set_data(out_ed[int(idx)])
    p.set_data(in_sphere[int(idx)])
    o.set_data(out_sphere[int(idx)])
    k.set_data(points_1[int(idx)])
    fig1.canvas.draw_idle()
slidx.on_changed(update)

plt.show()