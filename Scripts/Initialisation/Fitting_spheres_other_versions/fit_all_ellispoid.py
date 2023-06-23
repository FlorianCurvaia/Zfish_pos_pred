#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:38:10 2023

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

import jax.numpy as jnp
from jax import grad
from jax import random
from jax.config import config
config.update('jax_enable_x64', True)
key = random.PRNGKey(0)


#fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B08_px+1076_py-0189.h5'
#fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B07_px+1257_py-0474.h5'
fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B02_px-0280_py+1419.h5'


start_time_0=time.time()
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'embryo'}):
        seg_emb = to_numpy(dset)
        
print("h5_select --- %s seconds ---\n" % (time.time() - start_time_0))

z_width_low=math.floor(seg_emb.shape[0]*0.05)
z_width_high=seg_emb.shape[0]-math.floor(seg_emb.shape[0]*0.1)


start_time_1=time.time()
mask=np.copy(seg_emb)
seg_emb[~mask] = 0
print("doing mask --- %s seconds ---\n" % (time.time() - start_time_1))

#Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays
start_time_2=time.time()
struct = ndimage.generate_binary_structure(3, 3)
erode = ndimage.binary_erosion(mask, struct)
edges = mask ^ erode


print("erosion --- %s seconds ---\n" % (time.time() - start_time_2))
del mask, erode, struct, seg_emb

#Source:https://jekel.me/2020/Least-Squares-Ellipsoid-Fit/
def ellipsoidFit(coords):
    #   Assemble the A matrix
    elliX = np.squeeze(coords[1])
    elliY = np.squeeze(coords[2])
    elliZ = np.squeeze(coords[0])
    A = np.array([elliX**2, elliY**2, elliZ**2]).T
    
    O = np.ones(len(elliX))

    #   Assemble the f matrix
    B, resids, rank, s = np.linalg.lstsq(A, O)
    a_ls = np.sqrt(1.0/B[0])
    b_ls = np.sqrt(1.0/B[1])
    c_ls = np.sqrt(1.0/B[2])

    return a_ls, b_ls, c_ls

start_time_3=time.time()
tu=np.unravel_index(np.where(edges.ravel()==True),edges.shape)
idx=np.random.randint(len(np.squeeze(tu[0])), size=500)
tu_1=(tu[0][:,idx],tu[2][:,idx], tu[1][:,idx])

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

r, x0, y0, z0=sphereFit(tu_1)

gamma_guess = np.random.random(6)
gamma_guess[0] = x0
gamma_guess[1] = y0
gamma_guess[2] = z0
gamma_guess[3] = r
gamma_guess[4] = r
gamma_guess[5] = np.max(tu_1[0])-np.min(tu_1[0])
gamma_guess = jnp.array(gamma_guess)

def predict(gamma):
    # compute f hat
    x0 = gamma[0]
    y0 = gamma[1]
    z0 = gamma[2]
    a2 = gamma[3]**2
    b2 = gamma[4]**2
    c2 = gamma[5]**2
    zeta0 = (x - x0)**2 / a2
    zeta1 = (y - y0)**2 / b2
    zeta2 = (z - z0)**2 / c2
    return zeta0 + zeta1 + zeta2


def loss(g):
    # compute mean squared error
    pred = predict(g)
    target = jnp.ones_like(pred)
    mse = jnp.square(pred-target).mean()
    return mse


print(loss(gamma_guess))

a, b, c = ellipsoidFit(tu_1)
print("fitting middle sphere--- %s seconds ---\n" % (time.time() - start_time_3))
z, y, x = np.ogrid[0:len(edges), 0:1000, 0:1000]
mid_sphere = np.add(np.add(np.square(x-x0), np.square(y-y0)), np.square(z-z0)) <=r**2
print("get middle sphere--- %s seconds ---\n" % (time.time() - start_time_3))
del tu, idx, tu_1, z, y, x

mask=np.copy(mid_sphere)
mid_sphere[~mask] = 0

#Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays

struct = ndimage.generate_binary_structure(3, 3)
erode = ndimage.binary_erosion(mask, struct)
mid_sphere_ed = mask ^ erode

np.save("Arrays/mid_sphere_B02_px-0280_py+1419", mid_sphere_ed)
np.save("Arrays/edges_B02_px-0280_py+1419", edges)


start_time_4=time.time()

out_edges=np.copy(edges)
out_edges[mid_sphere]=False
del mid_sphere, mask, erode, struct


#tu_2=np.unravel_index(np.where(out_edges.ravel()==True),out_edges.shape)

num_to_sample_slice=math.floor(5000/len(out_edges))
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
    
tu=(idx_z_2, idx_x_2, idx_y_2)


r_2, x0_2, y0_2, z0_2 = sphereFit(tu)
print("refining sphere--- %s seconds ---\n" % (time.time() - start_time_4))
z, y, x = np.ogrid[0:len(out_edges), 0:1000, 0:1000]
out_sphere = np.add(np.add(np.square(x-x0_2), np.square(y-y0_2)), np.square(z-z0_2)) <=r_2**2
print("get better sphere--- %s seconds ---\n" % (time.time() - start_time_4))

mask=np.copy(out_sphere)
out_sphere[~mask] = 0

#Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays

struct = ndimage.generate_binary_structure(3, 3)
erode = ndimage.binary_erosion(mask, struct)
out_sphere_ed = mask ^ erode

del mask, erode, struct

np.save("Arrays/out_sphere_B02_px-0280_py+1419", out_sphere_ed)
np.save("Arrays/out_edges_B02_px-0280_py+1419", out_edges)
np.save("Arrays/out_sphere_origin_B02_px-0280_py+1419", np.array([x0_2[0], y0_2[0], z0_2[0]]))

start_time_5=time.time()
points_out=np.zeros((out_sphere.shape))
points_out[tu[0], tu[1], tu[2]]=True
sum_2D_points_out=[]
for i in range(len(points_out)):
    sum_2D_points_out.append(np.sum(points_out[i]))
fig1=plt.figure(1)
plt.clf()
plt.plot(range(len(points_out)), sum_2D_points_out)
r = 10

xx, yy = np.mgrid[-r:r+1, -r:r+1]
circle = xx**2 + yy**2 <= r**2

#Source: https://stackoverflow.com/questions/71875073/draw-a-circle-in-a-numpy-array-given-index-and-radius-without-external-libraries
for point in points_out:
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

del idx_z_2, idx_x_2, idx_y_2, tu, z, y, x, out_sphere

###GET INNER SPHERE

start_time_4=time.time()
in_ed=(edges.astype(np.float32) - out_edges.astype(np.float32)).astype(bool)
in_edges=in_ed[0:int(np.floor(0.9*in_ed.shape[0]))]
#tu_2=np.unravel_index(np.where(in_edges.ravel()==True),in_edges.shape)
num_to_sample_slice=math.floor(5000/in_edges.shape[0])
idx_x_2=np.empty((0,0), dtype="int64")
idx_y_2=np.empty((0,0), dtype="int64")
idx_z_2=np.empty((0,0), dtype="int64")
for sli in range(1,len(in_edges)-1):
    tot_sli=np.sum(in_edges[sli])
    if num_to_sample_slice>tot_sli:
        if tot_sli==0:
            pass
        elif tot_sli==1:
            tu_sli=np.unravel_index(np.where(in_edges[sli].ravel()==True),in_edges[sli].shape)
            tu_sli_samp=(tu_sli[1][:,0], tu_sli[0][:,0])
            tu_sli_z = np.full((1, tot_sli), sli)
            idx_x_2=np.concatenate((idx_x_2,tu_sli_samp[0]), axis=None)
            idx_y_2=np.concatenate((idx_y_2,tu_sli_samp[1]), axis=None)
            idx_z_2=np.concatenate((idx_z_2, tu_sli_z), axis=None)
            
        else:
            tu_sli=np.unravel_index(np.where(in_edges[sli].ravel()==True),in_edges[sli].shape)
            to_samp=np.random.randint(len(np.squeeze(tu_sli[0])), size=tot_sli)
            tu_sli_samp=(tu_sli[1][:,to_samp], tu_sli[0][:,to_samp])
            tu_sli_z = np.full((1, tot_sli), sli)
            idx_x_2=np.concatenate((idx_x_2,tu_sli_samp[0]), axis=None)
            idx_y_2=np.concatenate((idx_y_2,tu_sli_samp[1]), axis=None)
            idx_z_2=np.concatenate((idx_z_2, tu_sli_z), axis=None)
    else:
        tu_sli=np.unravel_index(np.where(in_edges[sli].ravel()==True),in_edges[sli].shape)
        to_samp=np.random.randint(len(np.squeeze(tu_sli[0])), size=num_to_sample_slice)
        tu_sli_samp=(tu_sli[1][:,to_samp], tu_sli[0][:,to_samp])
        tu_sli_z = np.full((1, num_to_sample_slice), sli)
        idx_x_2=np.concatenate((idx_x_2,tu_sli_samp[0]), axis=None)
        idx_y_2=np.concatenate((idx_y_2,tu_sli_samp[1]), axis=None)
        idx_z_2=np.concatenate((idx_z_2, tu_sli_z), axis=None)
    
tu=(idx_z_2, idx_x_2, idx_y_2)


r_2, x0_2, y0_2, z0_2 = sphereFit(tu)
print("fitting inner sphere--- %s seconds ---\n" % (time.time() - start_time_4))
z, y, x = np.ogrid[0:len(in_ed), 0:1000, 0:1000]
in_sphere = np.add(np.add(np.square(x-x0_2), np.square(y-y0_2)), np.square(z-z0_2)) <=r_2**2
print("get inner sphere--- %s seconds ---\n" % (time.time() - start_time_4))
points_in=np.zeros((in_ed.shape))
points_in[tu[0], tu[1], tu[2]]=True
sum_2D_points_in=[]
for i in range(len(points_in)):
    sum_2D_points_in.append(np.sum(points_in[i]))
fig0=plt.figure(0)
plt.clf()
plt.plot(range(len(points_in)), sum_2D_points_in)

mask=np.copy(in_sphere)
in_sphere[~mask] = 0

#Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays

struct = ndimage.generate_binary_structure(3, 3)
erode = ndimage.binary_erosion(mask, struct)
in_sphere_ed = mask ^ erode
np.save("Arrays/in_sphere_B02_px-0280_py+1419", in_sphere_ed)



del idx_z_2, idx_x_2, idx_y_2, tu, z, y, x, in_sphere, mask, erode, struct

fig2=plt.figure(2)
plt.clf()
idx0 = 3
m = plt.imshow(edges[idx0], vmin=0, vmax=1)
#q = plt.imshow(out_edges[idx0], vmin=0, vmax=1)
#l = plt.imshow(seg_emb[idx0], vmin=0, vmax=1, alpha=0.5)
#o = plt.imshow(points_out[idx0], vmin=0, vmax=1, alpha=0.5)
#n = plt.imshow(mid_sphere_ed[idx0], vmin=0, vmax=1, alpha=0.35)
#p = plt.imshow(out_sphere_ed[idx0], vmin=0, vmax=1, alpha=0.25)
#k = plt.imshow(in_sphere_ed[idx0], vmin=0, vmax=1, alpha=0.15)

axidx = plt.axes([0.25, 0, 0.65, 0.03])
slidx = Slider(axidx, 'z-stack', 0, edges.shape[0]-1, valinit=idx0, valfmt='%d')

def update(val):
    idx = slidx.val
    m.set_data(edges[int(idx)])
    #q.set_data(edges_2[int(idx)])
    #l.set_data(seg_emb[int(idx)])
    #k.set_data(in_sphere_ed[int(idx)])
    #o.set_data(points_out[int(idx)])
    #n.set_data(mid_sphere_ed[int(idx)])
    #p.set_data(out_sphere_ed[int(idx)])
    fig1.canvas.draw_idle()
slidx.on_changed(update)

plt.show()