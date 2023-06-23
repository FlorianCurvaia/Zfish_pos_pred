#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 08:42:59 2023

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


#fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B08_px+1076_py-0189.h5'
fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B07_px+1257_py-0474.h5'

start_time_0=time.time()
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'embryo'}):
        seg_emb = to_numpy(dset)
        
print("h5_select --- %s seconds ---\n" % (time.time() - start_time_0))

z_width_low=math.floor(seg_emb.shape[0]*0.05)
z_width_high=seg_emb.shape[0]-math.floor(seg_emb.shape[0]*0.1)
#seg_emb_1=np.copy(seg_emb[math.floor(seg_emb.shape[0]*0.5):seg_emb.shape[0]-math.floor(seg_emb.shape[0]*0.125)])
#del seg_emb

start_time_1=time.time()
#mask = x**2 + y**2 + z**2 <= 500**2
mask=np.copy(seg_emb)
seg_emb[~mask] = 0
print("doing mask --- %s seconds ---\n" % (time.time() - start_time_1))

#Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays
start_time_2=time.time()
struct = ndimage.generate_binary_structure(3, 3)
erode = ndimage.binary_erosion(mask, struct)
edges = mask ^ erode
#edges_2 = ndimage.binary_erosion(mask)

#edge_idx = np.vstack(np.where(edges)).T

print("erosion --- %s seconds ---\n" % (time.time() - start_time_2))
del mask, erode, struct

#Source:https://jekel.me/2015/Least-Squares-Sphere-Fit/
#def sphereFit(spX,spY,spZ): 
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

start_time_3=time.time()
#tu=np.unravel_index(np.where(edges[z_width_low:z_width_high].ravel()==True),edges[z_width_low:z_width_high].shape)
tu=np.unravel_index(np.where(edges.ravel()==True),edges.shape)
idx=np.random.randint(len(np.squeeze(tu[0])), size=500)
tu_1=(tu[0][:,idx],tu[2][:,idx], tu[1][:,idx])
r, x0, y0, z0 = sphereFit(tu_1)
print("fitting middle sphere--- %s seconds ---\n" % (time.time() - start_time_3))
z, y, x = np.ogrid[0:len(seg_emb), 0:1000, 0:1000]
mid_sphere = np.add(np.add(np.square(x-x0), np.square(y-y0)), np.square(z-z0)) <=r**2
print("get middle sphere--- %s seconds ---\n" % (time.time() - start_time_3))
del tu, idx, tu_1, z, y, x

mask=np.copy(mid_sphere)
mid_sphere[~mask] = 0

#Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays

struct = ndimage.generate_binary_structure(3, 3)
erode = ndimage.binary_erosion(mask, struct)
mid_sphere_ed = mask ^ erode

np.save("Arrays/mid_sphere_B07_px+1257_py-0474", mid_sphere_ed)
np.save("Arrays/edges_B07_px+1257_py-0474", edges)


start_time_4=time.time()

out_edges=np.copy(edges)
out_edges[mid_sphere]=False
del mid_sphere

#edges_2=np.load("Arrays/edges_2_2.npy")
#tu_2=np.unravel_index(np.where(edges_2[z_width_low:z_width_high].ravel()==True),edges_2[z_width_low:z_width_high].shape)
tu_2=np.unravel_index(np.where(out_edges.ravel()==True),out_edges.shape)
"""
ed_tot_2=tu_2[0].shape[1]
incr_2=math.floor(ed_tot_2/5000)
idx_2=[]
for i in range(0, ed_tot_2, incr_2):
    idx_2.append(i)
"""
num_to_sample_slice=math.floor(5000/len(seg_emb))
idx_x_2=np.empty((0,0), dtype="int64")
idx_y_2=np.empty((0,0), dtype="int64")
idx_z_2=np.empty((0,0), dtype="int64")
for sli in range(1,len(edges)-1):
    tot_sli=np.sum(out_edges[sli])
    if num_to_sample_slice>tot_sli:
        if tot_sli==0:
            pass
        elif tot_sli==1:
            tu_sli=np.unravel_index(np.where(out_edges[sli].ravel()==True),out_edges[sli].shape)
            #to_samp=np.random.randint(len(np.squeeze(tu_sli[0])), size=tot_sli)
            tu_sli_samp=(tu_sli[1][:,0], tu_sli[0][:,0])
            tu_sli_z = np.full((1, tot_sli), sli)
            idx_x_2=np.concatenate((idx_x_2,tu_sli_samp[0]), axis=None)
            idx_y_2=np.concatenate((idx_y_2,tu_sli_samp[1]), axis=None)
            idx_z_2=np.concatenate((idx_z_2, tu_sli_z), axis=None)
        else:
            tu_sli=np.unravel_index(np.where(out_edges[sli].ravel()==True),out_edges[sli].shape)
            to_samp=np.random.randint(len(np.squeeze(tu_sli[0])), size=tot_sli)
            tu_sli_samp=(tu_sli[1][:,to_samp], tu_sli[0][:,to_samp])
            tu_sli_z = np.full((1, tot_sli), sli)
            idx_x_2=np.concatenate((idx_x_2,tu_sli_samp[0]), axis=None)
            idx_y_2=np.concatenate((idx_y_2,tu_sli_samp[1]), axis=None)
            idx_z_2=np.concatenate((idx_z_2, tu_sli_z), axis=None)
    else:
        tu_sli=np.unravel_index(np.where(out_edges[sli].ravel()==True),out_edges[sli].shape)
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
print("refining sphere--- %s seconds ---\n" % (time.time() - start_time_4))
z, y, x = np.ogrid[0:len(seg_emb), 0:1000, 0:1000]
out_sphere = np.add(np.add(np.square(x-x0_2), np.square(y-y0_2)), np.square(z-z0_2)) <=r_2**2
print("get better sphere--- %s seconds ---\n" % (time.time() - start_time_4))

mask=np.copy(out_sphere)
out_sphere[~mask] = 0

#Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays

struct = ndimage.generate_binary_structure(3, 3)
erode = ndimage.binary_erosion(mask, struct)
out_sphere_ed = mask ^ erode

np.save("Arrays/out_sphere_B07_px+1257_py-0474", out_sphere_ed)
np.save("Arrays/out_edges_B07_px+1257_py-0474", out_edges)
np.save("Arrays/out_sphere_origin_B07_px+1257_py-0474", np.array([x0_2[0], y0_2[0], z0_2[0]]))

start_time_5=time.time()
points=np.zeros((out_sphere.shape))
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

#Source: https://stackoverflow.com/questions/71875073/draw-a-circle-in-a-numpy-array-given-index-and-radius-without-external-libraries
for point in points:
    condition = np.where(point==True) # np.where(arr > .5) to benchmark 60k circles
    for x,y in zip(*condition):
        # valid indices of the array
        i = slice(max(x-r,0), min(x+r+1, point.shape[1]))
        j = slice(max(y-r,0), min(y+r+1, point.shape[0]))
    
        # visible slice of the circle
        ci = slice(abs(min(x-r, 0)), circle.shape[1] - abs(min(point.shape[1]-(x+r+1), 0)))
        cj = slice(abs(min(y-r, 0)), circle.shape[0] - abs(min(point.shape[0]-(y+r+1), 0)))
        
        point[j, i] += circle[cj, ci]
print("get circle on chosen points--- %s seconds ---\n" % (time.time() - start_time_5))

#del tu_2,idx_2, tu_3, z, y, x
del idx_z_2, idx_x_2, idx_y_2, tu_3, z, y, x, out_sphere

fig1=plt.figure(1)
plt.clf()
idx0 = 3
m = plt.imshow(edges[idx0], vmin=0, vmax=1)
#q = plt.imshow(out_edges[idx0], vmin=0, vmax=1)
#k = plt.imshow(mask[idx0], vmin=0, vmax=1, alpha=0.75)
#l = plt.imshow(seg_emb[idx0], vmin=0, vmax=1, alpha=0.5)
o = plt.imshow(points[idx0], vmin=0, vmax=1, alpha=0.5)
n = plt.imshow(mid_sphere_ed[idx0], vmin=0, vmax=1, alpha=0.35)
p = plt.imshow(out_sphere_ed[idx0], vmin=0, vmax=1, alpha=0.15)

axidx = plt.axes([0.25, 0, 0.65, 0.03])
slidx = Slider(axidx, 'z-stack', 0, edges.shape[0]-1, valinit=idx0, valfmt='%d')

def update(val):
    idx = slidx.val
    m.set_data(edges[int(idx)])
    #q.set_data(edges_2[int(idx)])
    #l.set_data(seg_emb[int(idx)])
    #k.set_data(mask[int(idx)])
    o.set_data(points[int(idx)])
    n.set_data(mid_sphere_ed[int(idx)])
    p.set_data(out_sphere_ed[int(idx)])
    fig1.canvas.draw_idle()
slidx.on_changed(update)

plt.show()