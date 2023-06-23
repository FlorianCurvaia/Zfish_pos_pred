#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 15:31:17 2023

@author: floriancurvaia
"""


import h5py

import napari

from abbott.h5_files import *

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, CheckButtons

from pathlib import Path

import math

import random

import time

def sphereFit(coords):
    #   Assemble the A matrix
    spX = np.squeeze(coords[2])#*0.65
    spY = np.squeeze(coords[1])#*0.65
    spZ = np.squeeze(coords[0])
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f, rcond=None)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2] 

def get_middle_sphere(edges, n_points=5000):
    random.seed(42)
    rand_id_z, rand_id_x, rand_id_y=np.unravel_index(
        random.sample(
            np.where(edges.ravel()==True)[0].tolist(),
            k=n_points), 
        edges.shape)
    indices=(rand_id_z, rand_id_y, rand_id_x)
    r, x0, y0, z0 = sphereFit(indices)
    return (r, x0, y0, z0)


#D05 weird
im="D05_px-0738_py-1248" 
#B07_px+1257_py-0474 B08_px+1076_py-0189 C04_px-0816_py-1668 B03_px-0545_py-1946 B06_px-0493_py+0146 C05_px+0198_py+1683
#D06_px-1055_py-0118 D05_px-0738_py-1248 B05_px+1522_py-1087

#fn_in='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/'+im+".h5"

fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/tmp_fcurvaia/"+im+".h5"

#fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Compressed_files/'+im+".h5"

fn_out='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Compressed_files/'+im+"_shifted.h5"

fn_in='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Compressed_aligned/Cluster/'+im+"_shifted.h5" #ok_trunc_975/

#fn_in='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Compressed_aligned/Old/'+im+"_shifted_try_1.h5"

#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/results/B04_px-0190_py+2329.h5"

"""
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'nucleiRaw'}):
        Nucleiseg = to_numpy(dset)
        #print(dset)

with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'embryo'}):
        emb_seg = to_numpy(dset)
        #print(dset)

with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'cells'}):
        cell_seg = to_numpy(dset)
      

cycle=0

with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'betaCatenin', 'level': 1, 'cycle': cycle}):
        cycle = dset.attrs['cycle']
        scale = dset.attrs['element_size_um']
        betaCatenin = to_numpy(dset)
        #betaCateninseg = np.multiply(to_numpy(dset), (Nucleiseg > 0))

with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'DAPI', 'level': 1, 'cycle': cycle}):
        DAPI_0 = to_numpy(dset)

cycle=1
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'DAPI', 'level': 1, 'cycle': cycle}):
        DAPI_1 = to_numpy(dset)
        

cycle=2
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'MapK', 'level': 1, 'cycle': cycle}):
        MapK = to_numpy(dset)
        
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'pSmad2/3', 'level': 1, 'cycle': cycle}):
        pSmad23 = to_numpy(dset)
        
cycle=4
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'pSmad1/5', 'level': 1, 'cycle': cycle}):
        pSmad15 = to_numpy(dset)

### put inv. commas below    
"""

cycle=2
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'DAPI', 'level': 1, 'cycle': cycle}):
        cycle = dset.attrs['cycle']
        scale = dset.attrs['element_size_um']
        DAPI_2 = to_numpy(dset)
        

cycle=4
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'DAPI', 'level': 1, 'cycle': cycle}):
        DAPI_4 = to_numpy(dset)#[40:]


#sphere=np.load(path_in_sph+im+".npy") #(path_in_sph+"out_sphere_"+im+".npy")

#pSmad15_shift=np.roll(pSmad15, -shift, axis=0)
#DAPI_4_shift=np.roll(DAPI_4, -shift, axis=0)
DAPI_4_trunc_0=DAPI_4[40:]
#DAPI_4_trunc=np.roll(DAPI_4_trunc_0, -25, 1)
#DAPI_4_trunc=np.roll(DAPI_4_trunc, -34, 2)
#DAPI_2[:, 975:, :]=0
DAPI_2_bool=DAPI_2>np.quantile(DAPI_2, 0.99)
DAPI_4_bool_bis=DAPI_4>np.quantile(DAPI_4, 0.99)

start_time_0=time.time()
r_4_bis, x_4_bis, y_4_bis, z_4_bis = get_middle_sphere(DAPI_4_bool_bis, 25000)
c_4_bis=np.array([x_4_bis, y_4_bis, z_4_bis])
print("Get sphere 4 bis --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()
r_2, x_2, y_2, z_2 = get_middle_sphere(DAPI_2_bool, 25000)
c_2=np.array([x_2, y_2, z_2])
print("Get sphere 2 --- %s seconds ---\n" % (time.time() - start_time_1))

print(np.linalg.norm(c_2-c_4_bis))

shift=math.ceil(np.abs(r_2-r_4_bis)+np.abs(z_2-z_4_bis))
print(shift)

#DAPI_4_trunc_0=DAPI_4[shift:]

DAPI_4_bool=DAPI_4_trunc_0>np.quantile(DAPI_4_trunc_0, 0.99)

r_4, x_4, y_4, z_4 = get_middle_sphere(DAPI_4_bool, 25000)


x_shift= int(math.floor(x_2[0]-x_4[0]))
y_shift= int(math.floor(y_2[0]-y_4[0]))

DAPI_4_trunc=np.roll(DAPI_4_trunc_0, x_shift, 1)
DAPI_4_trunc=np.roll(DAPI_4_trunc, y_shift, 2)


DAPI_4_bool=np.roll(DAPI_4_bool, x_shift, 1)
DAPI_4_bool=np.roll(DAPI_4_bool, y_shift, 2)

DAPI_2_trunc=DAPI_2.copy()
#mid_last_y=DAPI_4_trunc[-1][500, :]
#y_max=np.argmax(np.abs(np.diff(mid_last_y.astype("int64"))))
DAPI_2_trunc[:, :, -abs(y_shift):]=0

#mid_last_x=DAPI_4_trunc[-1][:, 500]
#x_max=np.argmax(np.abs(np.diff(mid_last_x.astype("int64"))))
DAPI_2_trunc[:, -abs(x_shift):, :]=0

#cut=max(y_max, x_max)+1
#DAPI_2[:, cut:, :]=0
#DAPI_2[:, :, cut:]=0

"""
with h5py.File(fn_out, "a") as f_out:
    with h5py.File(fn) as f_in:
        for dset in h5_select(f_in, {'stain': 'DAPI', 'level': 1, 'cycle': 4}):
            DAPI_4_attr=dset
        h5_write_channel(f_out, DAPI_4_trunc, "/ch_15/1", copy_from=DAPI_4_attr)
        for dset in h5_select(f_in, {'stain': 'DAPI', 'level': 1, 'cycle': 2}):
            DAPI_2_attr=dset
        h5_write_channel(f_out, DAPI_2_trunc, "/ch_08/1", copy_from=DAPI_2_attr)
        
fld=Path("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Compressed_files/")


fns = sorted(list(fld.glob('*.h5')))
print(fns)      
"""


"""     


   
with h5py.File(fn_in) as f:
   
    for dset in h5_select(f, {'stain': 'DAPI', 'level': 1, 'cycle': 4}):
        DAPI_4=to_numpy(dset)

    for dset in h5_select(f, {'stain': 'DAPI', 'level': 1, 'cycle': 2}):
        scale = dset.attrs['element_size_um']
        DAPI_2=to_numpy(dset)
 
   
### put inv. commas below   



fig1=plt.figure(1)
plt.clf()
idx0 = 3
#m = plt.imshow(out_elli[idx0], vmin=0, vmax=1)
k = plt.imshow(DAPI_2[idx0], alpha=1, cmap="Blues")
n = plt.imshow(DAPI_4_trunc_0[idx0], alpha=0.25, cmap="Greens")
p = plt.imshow(DAPI_4_trunc[idx0], alpha=0.25, cmap="Reds")

 #vmin=np.min(points), vmax=np.max(points), alpha=0.75)
#o = plt.imshow(out_sphere[idx0], vmin=0, vmax=1, alpha=0.25)

axidx = plt.axes([0.25, 0, 0.65, 0.03])
slidx = Slider(axidx, 'z-stack', 0, DAPI_2.shape[0]-1, valinit=idx0, valfmt='%d')

channels=[n, k, p]
rax = plt.axes([0.05, 0.4, 0.1, 0.15])
labels = [ "DAPI_4_trunc_0","DAPI_2", "DAPI_4_trunc"]
visibility = [chan.get_visible() for chan in channels]
check = CheckButtons(rax, labels, visibility)


def func(label):
    index = labels.index(label)
    channels[index].set_visible(not channels[index].get_visible())
    plt.draw()

check.on_clicked(func)

def update(val):
    idx = slidx.val
    #m.set_data(out_elli[int(idx)])
    n.set_data(DAPI_4_trunc_0[int(idx)])
    p.set_data(DAPI_4_trunc[int(idx)])
    #o.set_data(out_sphere[int(idx)])
    k.set_data(DAPI_2[int(idx)])
    fig1.canvas.draw_idle()
slidx.on_changed(update)

plt.show()
"""


viewer = napari.Viewer()

#Dist=np.load("min_dists_y_x.npy")
#image_layer=viewer.add_points(data=coords, ndim=3, properties=Dist)
#image_layer = viewer.add_image(DAPI_0, scale=scale, colormap='magenta', blending='additive',visible=False)
#image_layer = viewer.add_image(DAPI_1, scale=scale, colormap='magenta', blending='additive',visible=False)
image_layer = viewer.add_image(DAPI_2, scale=scale, colormap='blue', blending='additive',visible=False)
image_layer = viewer.add_image(DAPI_4, scale=scale, colormap='red', blending='additive',visible=False)

#image_layer = viewer.add_image(DAPI_4_shift, scale=scale, colormap='green', blending='additive',visible=False)
image_layer = viewer.add_image(DAPI_4_trunc_0, scale=scale, colormap='red', blending='additive',visible=False)
image_layer = viewer.add_image(DAPI_4_trunc, scale=scale, colormap='red', blending='additive',visible=False)
image_layer = viewer.add_image(DAPI_2_trunc, scale=scale, colormap='green', blending='additive',visible=False)

#image_layer = viewer.add_labels(DAPI_2_bool, scale=scale, blending='additive',visible=False)
#image_layer = viewer.add_labels(DAPI_4_bool, scale=scale, blending='additive',visible=False)






"""
with h5py.File(fn, "a") as f_out:
    for dset in h5_select(fn, {'stain': 'DAPI', 'level': 1, 'cycle': 4}):
        DAPI_4_to_attr=dset
        DAPI_4_attrs=DAPI_4_to_attr.attrs
        DAPI_4_attrs.create("stain",DAPI_4_attrs["stain"]+"_corr" )
    
    h5_write_channel(f_out, DAPI_4_trunc, "/ch_15/1", attrs=DAPI_4_attrs)
    for dset in h5_select(f_in, {'stain': 'DAPI', 'level': 1, 'cycle': 2}):
        DAPI_2_attr=dset
        DAPI_2_attrs=DAPI_2_to_attr.attrs
        DAPI_2_attrs.create("stain",DAPI_2_attrs["stain"]+"_corr" )
    h5_write_channel(f_out, DAPI_2, "/ch_08/1", attrs=DAPI_2_attr)

fld=Path("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/tmp_fcurvaia/")
"""

"""
start_time_2=time.time()
DAPI_4_trunc=np.pad(DAPI_4_trunc_0.astype("int64"), ((0, 0), (abs(x_shift), abs(x_shift)), (abs(y_shift), abs(y_shift))), 'constant', constant_values=(-1,))
DAPI_4_trunc=np.roll(DAPI_4_trunc, x_shift, 1)
DAPI_4_trunc=np.roll(DAPI_4_trunc, y_shift, 2)
DAPI_4_trunc=np.reshape(np.delete(DAPI_4_trunc, DAPI_4_trunc.flatten()==-1), DAPI_4_trunc_0.shape).astype("uint16")
print("Shift array--- %s seconds ---\n" % (time.time() - start_time_2))
"""

