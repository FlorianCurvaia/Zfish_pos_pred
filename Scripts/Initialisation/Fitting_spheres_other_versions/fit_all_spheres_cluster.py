#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:06:50 2023

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
fn=path_in+str(sys.argv[1])


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

np.save(path_out+"mid_sphere_B02_px-0280_py+1419", mid_sphere_ed)
np.save(path_out+"edges_B02_px-0280_py+1419", edges)


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

np.save(path_out+"out_sphere_B02_px-0280_py+1419", out_sphere_ed)
np.save(path_out+"out_edges_B02_px-0280_py+1419", out_edges)
np.save(path_out+"out_sphere_origin_B02_px-0280_py+1419", np.array([x0_2[0], y0_2[0], z0_2[0]]))


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


mask=np.copy(in_sphere)
in_sphere[~mask] = 0

#Source: https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays

struct = ndimage.generate_binary_structure(3, 3)
erode = ndimage.binary_erosion(mask, struct)
in_sphere_ed = mask ^ erode
np.save(path_out+"in_sphere_B02_px-0280_py+1419", in_sphere_ed)



del idx_z_2, idx_x_2, idx_y_2, tu, z, y, x, in_sphere, mask, erode, struct

