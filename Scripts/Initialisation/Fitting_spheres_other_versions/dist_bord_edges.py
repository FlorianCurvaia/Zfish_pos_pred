#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:46:02 2023

@author: floriancurvaia
"""

import numpy as np

#from math import sqrt, acos
import math 

import time

import h5py

from joblib import Parallel, delayed

from abbott.h5_files import *

from scipy.spatial.distance import cdist

import pandas as pd
from matplotlib.widgets import Slider
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from math import pi

import numba

import scipy.ndimage as ndimage

from scipy.signal import argrelextrema

start_time_0=time.time()

edges=np.load("Arrays/edges_B02_px-0280_py+1419.npy")
#edges_2=np.load("edges_2.npy")

print("Read and load files --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()
origin=np.load("Arrays/out_sphere_origin_B02_px-0280_py+1419.npy")
#origin2=[500,500,edges.shape[0]]

ed_id=np.array(np.where(edges==True))
#orgn=np.array(origin)

print("Prepare data --- %s seconds ---\n" % (time.time() - start_time_1))



# EUCLEDIAN DISTANCE Source: https://stackoverflow.com/questions/49664523/how-can-i-speed-up-closest-point-comparison-using-cdist-or-tensorflow
@numba.njit('(float64[::1], float64[::1], float64[::1], float64[::1])', parallel=True, fastmath=True)
def pz_dist(p_array, x_flat, y_flat, z_flat):
    #m = p_array.shape[0]
    n = x_flat.shape[0]
    d = np.empty(shape=(n), dtype=np.float64)
    for j in range(n):
        _x = x_flat[j] - p_array[0]
        _y = y_flat[j] - p_array[1]
        _z = z_flat[j] - p_array[2]
        _d = _x**2 + _y**2 + _z**2
        d[j] = _d
    return d

start_time_3=time.time()
dists_ed=np.sqrt(pz_dist(origin, ed_id[0].astype("float64"), ed_id[1].astype("float64"), ed_id[2].astype("float64")))
print("Get distances --- %s seconds ---\n" % (time.time() - start_time_3))

fig1=plt.figure(1)
plt.clf()
plt.hist(dists_ed, bins=300)

hist=np.histogram(dists_ed, bins=300)
b=np.where(np.logical_and(hist[1]>=475, hist[1]<=540))[0]
min_id=np.max(np.where(hist[0]==np.min(hist[0][b])))
min_r=hist[1][min_id]

to_select=np.array(np.where(dists_ed>min_r))
out_b=ed_id[:, np.squeeze(to_select)]
edges_2_2=np.zeros(edges.shape)
edges_2_2[out_b[0], out_b[1], out_b[2]]=1
edges_2_2=np.array(edges_2_2, dtype=bool)

to_select_2=np.array(np.where(dists_ed<min_r))
out_b_2=ed_id[:, np.squeeze(to_select_2)]
edges_2_2_2=np.zeros(edges.shape)
edges_2_2_2[out_b_2[0], out_b_2[1], out_b_2[2]]=1
edges_2_2_2=np.array(edges_2_2_2, dtype=bool)
#outer_surface=edges_2+edges_2_2 >-1
#inner_surface=





def sphereFit(coords):
    #   Assemble the A matrix
    spX = np.squeeze(coords[1])
    spY = np.squeeze(coords[2])
    spZ = np.squeeze(coords[0])
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]

start_time_4=time.time()
in_ed=edges_2_2_2[0:int(np.floor(0.9*edges_2_2_2.shape[0]))]
#tu_2=np.unravel_index(np.where(edges_2[z_width_low:z_width_high].ravel()==True),edges_2[z_width_low:z_width_high].shape)
tu_2=np.unravel_index(np.where(in_ed.ravel()==True),in_ed.shape)
"""
ed_tot_2=tu_2[0].shape[1]
incr_2=math.floor(ed_tot_2/5000)
idx_2=[]
for i in range(0, ed_tot_2, incr_2):
    idx_2.append(i)
"""
num_to_sample_slice=math.floor(5000/in_ed.shape[0])
idx_x_2=np.empty((0,0), dtype="int64")
idx_y_2=np.empty((0,0), dtype="int64")
idx_z_2=np.empty((0,0), dtype="int64")
for sli in range(1,len(in_ed)-1):
    tot_sli=np.sum(in_ed[sli])
    if num_to_sample_slice>tot_sli:
        if tot_sli==0:
            pass
        elif tot_sli==1:
            tu_sli=np.unravel_index(np.where(in_ed[sli].ravel()==True),in_ed[sli].shape)
            #to_samp=np.random.randint(len(np.squeeze(tu_sli[0])), size=tot_sli)
            tu_sli_samp=(tu_sli[1][:,0], tu_sli[0][:,0])
            tu_sli_z = np.full((1, tot_sli), sli)
            idx_x_2=np.concatenate((idx_x_2,tu_sli_samp[0]), axis=None)
            idx_y_2=np.concatenate((idx_y_2,tu_sli_samp[1]), axis=None)
            idx_z_2=np.concatenate((idx_z_2, tu_sli_z), axis=None)
            
        else:
            tu_sli=np.unravel_index(np.where(in_ed[sli].ravel()==True),in_ed[sli].shape)
            to_samp=np.random.randint(len(np.squeeze(tu_sli[0])), size=tot_sli)
            tu_sli_samp=(tu_sli[1][:,to_samp], tu_sli[0][:,to_samp])
            tu_sli_z = np.full((1, tot_sli), sli)
            idx_x_2=np.concatenate((idx_x_2,tu_sli_samp[0]), axis=None)
            idx_y_2=np.concatenate((idx_y_2,tu_sli_samp[1]), axis=None)
            idx_z_2=np.concatenate((idx_z_2, tu_sli_z), axis=None)
    else:
        tu_sli=np.unravel_index(np.where(in_ed[sli].ravel()==True),in_ed[sli].shape)
        to_samp=np.random.randint(len(np.squeeze(tu_sli[0])), size=num_to_sample_slice)
        tu_sli_samp=(tu_sli[1][:,to_samp], tu_sli[0][:,to_samp])
        tu_sli_z = np.full((1, num_to_sample_slice), sli)
        idx_x_2=np.concatenate((idx_x_2,tu_sli_samp[0]), axis=None)
        idx_y_2=np.concatenate((idx_y_2,tu_sli_samp[1]), axis=None)
        idx_z_2=np.concatenate((idx_z_2, tu_sli_z), axis=None)
    
tu_3=(idx_z_2, idx_x_2, idx_y_2)
#idx_2=np.random.randint(len(np.squeeze(tu_2[0])), size=5000)
#tu_3=(tu_2[0][:,idx_2],tu_2[2][:,idx_2], tu_2[1][:,idx_2])

r_2, x0_2, y0_2, z0_2 = sphereFit(tu_3)
print("fitting inner sphere--- %s seconds ---\n" % (time.time() - start_time_4))
z, y, x = np.ogrid[0:307, 0:1000, 0:1000]
in_sphere = np.add(np.add(np.square(x-x0_2), np.square(y-y0_2)), np.square(z-z0_2)) <=r_2**2
print("get inner sphere--- %s seconds ---\n" % (time.time() - start_time_4))
points=np.zeros((edges_2_2_2.shape))
points[tu_3[0], tu_3[1], tu_3[2]]=True
sum_2D_points=[]
sum_2D_edges=[]
for i in range(len(points)):
    sum_2D_points.append(np.sum(points[i]))
    sum_2D_edges.append(np.sum(edges[i]))
fig3=plt.figure(3)
plt.clf()
plt.plot(range(len(points)), sum_2D_points)

mask=np.copy(in_sphere)
in_sphere[~mask] = 0

#Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays

struct = ndimage.generate_binary_structure(3, 3)
erode = ndimage.binary_erosion(mask, struct)
in_sphere_ed = mask ^ erode
np.save("Arrays/in_sphere_B07_px+1257_py-0474", in_sphere_ed)

out_sphere_ed=np.load("Arrays/out_sphere_B07_px+1257_py-0474.npy")


fig0=plt.figure(0)
plt.clf()
idx0 = 3

m = plt.imshow(edges_2_2[idx0], vmin=0, vmax=1, cmap="viridis")
n = plt.imshow(edges_2_2_2[idx0], vmin=0, vmax=1, cmap="magma", alpha=0.75)
#p = plt.imshow(in_sphere[idx0], vmin=0, vmax=1, cmap="viridis", alpha=0.5)
p = plt.imshow(in_sphere_ed[idx0], vmin=0, vmax=1, cmap="viridis", alpha=0.5)
k = plt.imshow(out_sphere_ed[idx0], vmin=0, vmax=1, cmap="viridis", alpha=0.25)

axidx = plt.axes([0.25, 0, 0.65, 0.03])
slidx = Slider(axidx, 'z-stack', 0, edges_2_2.shape[0]-1, valinit=idx0, valfmt='%d')

def update_1(val):
    idx = slidx.val
    m.set_data(edges_2_2[int(idx)])
    n.set_data(edges_2_2_2[int(idx)])
    p.set_data(in_sphere_ed[int(idx)])
    k.set_data(out_sphere_ed[int(idx)])
    fig0.canvas.draw_idle()
slidx.on_changed(update_1)
plt.show()
