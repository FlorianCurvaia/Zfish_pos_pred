#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:00:10 2023

@author: floriancurvaia
"""

import sys
import h5py
import numpy as np
import math
from abbott.h5_files import *
import time
import scipy.ndimage as ndimage



#fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B08_px+1076_py-0189.h5'
#fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B07_px+1257_py-0474.h5'
path_in='/data/active/sshami/20220716_experiment3_aligned/'
path_out="/data/homes/fcurvaia/Spheres_fit/"
fn=path_in+str(sys.argv[1])+".h5"
im=str(sys.argv[1])

start_time_0=time.time()
with h5py.File(fn) as f:
    seg_emb=np.array(f["lbl_embryo"])
        
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
def ellipsoidFit(coords, x_0, y_0, z_0):
    #   Assemble the A matrix
    elliX = np.squeeze(coords[1])-x_0
    elliY = np.squeeze(coords[2])-y_0
    elliZ = np.squeeze(coords[0])-z_0
    A = np.array([elliX**2, elliY**2, elliZ**2]).T
    
    O = np.ones(len(elliX))

    #   Assemble the f matrix
    B, resids, rank, s = np.linalg.lstsq(A, O)
    a_ls = np.sqrt(1.0/B[0])
    b_ls = np.sqrt(1.0/B[1])
    c_ls = np.sqrt(1.0/B[2])

    return a_ls, b_ls, c_ls


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
tu=np.unravel_index(np.where(edges.ravel()==True),edges.shape)
idx=np.random.randint(len(np.squeeze(tu[0])), size=500)
tu_1=(tu[0][:,idx],tu[2][:,idx], tu[1][:,idx])
r, x0, y0, z0 = sphereFit(tu_1)
a, b, c = ellipsoidFit(tu_1, x0, y0, z0)
print("fitting middle ellipsoid--- %s seconds ---\n" % (time.time() - start_time_3))
z, y, x = np.ogrid[0:len(edges), 0:1000, 0:1000]
mid_elli=np.add(np.add(np.square(x-x0)/a**2, np.square(y-y0)/b**2), np.square(z-z0)/c**2) <=1
#mid_elli = np.add(np.add(np.square(x-res[0][0])/res[0][3]**2, np.square(y-res[0][1])/res[0][4]**2), np.square(z-res[0][2])/res[0][5]**2) <=1
print("get middle ellipsoid--- %s seconds ---\n" % (time.time() - start_time_3))
del tu, idx, tu_1, z, y, x

mask=np.copy(mid_elli)
mid_elli[~mask] = 0

#Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays

struct = ndimage.generate_binary_structure(3, 3)
erode = ndimage.binary_erosion(mask, struct)
mid_elli_ed = mask ^ erode

np.save(path_out+"mid_elli_"+im, mid_elli_ed)
np.save(path_out+"edges_"+im, edges)


start_time_4=time.time()

out_edges=np.copy(edges)
out_edges[mid_elli]=False
del mid_elli, mask, erode, struct


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
a, b, c = ellipsoidFit(tu, x0_2, y0_2, z0_2)
print("fitting outer ellipsoid--- %s seconds ---\n" % (time.time() - start_time_4))
z, y, x = np.ogrid[0:len(edges), 0:1000, 0:1000]
out_elli=np.add(np.add(np.square(x-x0_2)/a**2, np.square(y-y0_2)/b**2), np.square(z-z0_2)/c**2) <=1
#mid_elli = np.add(np.add(np.square(x-res[0][0])/res[0][3]**2, np.square(y-res[0][1])/res[0][4]**2), np.square(z-res[0][2])/res[0][5]**2) <=1
print("get outer ellipsoid--- %s seconds ---\n" % (time.time() - start_time_4))


mask=np.copy(out_elli)
out_elli[~mask] = 0

#Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays

struct = ndimage.generate_binary_structure(3, 3)
erode = ndimage.binary_erosion(mask, struct)
out_elli_ed = mask ^ erode

del mask, erode, struct

np.save(path_out+"out_elli_"+im, out_elli_ed)
np.save(path_out+"out_edges_"+im, out_edges)
np.save(path_out+"out_elli_origin_"+im, np.array([x0_2[0], y0_2[0], z0_2[0]]))


del idx_z_2, idx_x_2, idx_y_2, tu, z, y, x, out_elli

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
a, b, c = ellipsoidFit(tu, x0_2, y0_2, z0_2)
print("fitting inner ellipsoid--- %s seconds ---\n" % (time.time() - start_time_4))
z, y, x = np.ogrid[0:len(edges), 0:1000, 0:1000]
in_elli=np.add(np.add(np.square(x-x0_2)/a**2, np.square(y-y0_2)/b**2), np.square(z-z0_2)/c**2) <=1
#mid_elli = np.add(np.add(np.square(x-res[0][0])/res[0][3]**2, np.square(y-res[0][1])/res[0][4]**2), np.square(z-res[0][2])/res[0][5]**2) <=1
print("get inner ellipsoid--- %s seconds ---\n" % (time.time() - start_time_4))


mask=np.copy(in_elli)
in_elli[~mask] = 0

#Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays

struct = ndimage.generate_binary_structure(3, 3)
erode = ndimage.binary_erosion(mask, struct)
in_elli_ed = mask ^ erode
np.save(path_out+"in_elli_"+im, in_elli_ed)



del idx_z_2, idx_x_2, idx_y_2, tu, z, y, x, in_elli, mask, erode, struct

