#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:32:08 2023

@author: floriancurvaia
"""
import h5py
import numpy as np
import math
from abbott.h5_files import *
import time
import scipy.ndimage as ndimage
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

#fit_sphere.py

"""
num_to_sample_slice=math.floor(5000/z_width)
idx_x=np.empty((0,0), dtype="int64")
idx_y=np.empty((0,0), dtype="int64")
idx_z=np.empty((0,0), dtype="int64")
for sli in range(1,len(edges)-1):
    tu_sli=np.unravel_index(np.where(edges[sli].ravel()==True),edges[sli].shape)
    tu_sli_z = np.full((1, tu_sli[0].shape[1]), sli)
    idx_x=
    
tu_sli=tuple(tu_sl_l)
"""

"""
ed_tot=tu[0].shape[1]
incr=math.floor(ed_tot/5000)
idx=[]
for i in range(0, ed_tot, incr):
    idx.append(i)
"""

"""
ed_tot_2=tu_2[0].shape[1]
incr_2=math.floor(ed_tot_2/5000)
idx_2=[]
for i in range(0, ed_tot_2, incr_2):
    idx_2.append(i)
"""

"""
start_time_5=time.time()
points=np.zeros((sphere.shape))
points[tu_1[0], tu_1[1], tu_1[2]]=True
sum_2D_points=[]
sum_2D_edges=[]
for i in range(len(points)):
    sum_2D_points.append(np.sum(points[i]))
    sum_2D_edges.append(np.sum(edges[i]))
fig2=plt.figure(2)
plt.clf()
plt.plot(range(len(points)), sum_2D_points)
#plt.plot(range(len(points)), sum_2D_edges)

r = 10

xx, yy = np.mgrid[-r:r+1, -r:r+1]
circle = xx**2 + yy**2 <= r**2

for point in points:
    condition = np.where(point==True) # np.where(arr > .5) to benchmark 60k circles
    for x,y in zip(*condition):
        # valid indices of the array
        i = slice(max(x-r,0), min(x+r+1, point.shape[0]))
        j = slice(max(y-r,0), min(y+r+1, point.shape[1]))
    
        # visible slice of the circle
        ci = slice(abs(min(x-r, 0)), circle.shape[0] - abs(min(point.shape[0]-(x+r+1), 0)))
        cj = slice(abs(min(y-r, 0)), circle.shape[1] - abs(min(point.shape[1]-(y+r+1), 0)))
        
        point[i, j] += circle[ci, cj]
print("get circle on chosen points--- %s seconds ---\n" % (time.time() - start_time_5))
"""


"""
start_time_5=time.time()
points=np.zeros((sphere_2.shape))
points[tu_3[0], tu_3[1], tu_3[2]]=True
sum_2D_points=[]
sum_2D_edges=[]
for i in range(len(points)):
    sum_2D_points.append(np.sum(points[i]))
    sum_2D_edges.append(np.sum(edges[i]))
fig3=plt.figure(3)
plt.clf()
plt.plot(range(len(points)), sum_2D_points)
r = 10

xx, yy = np.mgrid[-r:r+1, -r:r+1]
circle = xx**2 + yy**2 <= r**2

for point in points:
    condition = np.where(point==True) # np.where(arr > .5) to benchmark 60k circles
    for x,y in zip(*condition):
        # valid indices of the array
        i = slice(max(x-r,0), min(x+r+1, point.shape[0]))
        j = slice(max(y-r,0), min(y+r+1, point.shape[1]))
    
        # visible slice of the circle
        ci = slice(abs(min(x-r, 0)), circle.shape[0] - abs(min(point.shape[0]-(x+r+1), 0)))
        cj = slice(abs(min(y-r, 0)), circle.shape[1] - abs(min(point.shape[1]-(y+r+1), 0)))
        
        point[i, j] += circle[ci, cj]
print("get circle on chosen points--- %s seconds ---\n" % (time.time() - start_time_5))
"""


