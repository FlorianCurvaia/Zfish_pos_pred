#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:17:01 2023

@author: floriancurvaia
"""

import h5py

import napari

from abbott.h5_files import *

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, CheckButtons

from pathlib import Path

shift=40

im="C04_px-0816_py-1668" #B07_px+1257_py-0474 B08_px+1076_py-0189 C04_px-0816_py-1668 B03_px-0545_py-1946

#fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/'+im+".h5"

fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Compressed_files/'+im+".h5"

fn_out='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Compressed_aligned/'+im+"_shifted.h5" #ok_trunc_975/

#fn_out='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Compressed_files/'+im+"_shifted.h5"




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
DAPI_4_shift=np.roll(DAPI_4, -shift, axis=0)
DAPI_4_trunc_0=DAPI_4[40:]
DAPI_4_trunc=np.roll(DAPI_4_trunc_0, -25, 1)
DAPI_4_trunc=np.roll(DAPI_4_trunc, -34, 2)
#DAPI_2[:, 975:, :]=0


 
with h5py.File(fn_out, "a") as f_out:
    with h5py.File(fn) as f_in:
        for dset in h5_select(f_in, {'stain': 'DAPI', 'level': 1, 'cycle': 4}):
            DAPI_4_attr=dset
        h5_write_channel(f_out, DAPI_4_trunc_0, "/ch_15/1", copy_from=DAPI_4_attr)
        for dset in h5_select(f_in, {'stain': 'DAPI', 'level': 1, 'cycle': 2}):
            DAPI_2_attr=dset
        h5_write_channel(f_out, DAPI_2, "/ch_08/1", copy_from=DAPI_2_attr)
        

fld=Path("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Compressed_files/")

fns = sorted(list(fld.glob('*.h5')))
print(fns)      
        

"""     
   
with h5py.File(fn_out) as f:
   
    for dset in h5_select(f, {'stain': 'DAPI', 'level': 1, 'cycle': 4}):
        DAPI_4=to_numpy(dset)

    for dset in h5_select(f, {'stain': 'DAPI', 'level': 1, 'cycle': 2}):
        scale = dset.attrs['element_size_um']
        DAPI_2=to_numpy(dset)

   
### put inv. commas below   
"""     

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
labels = ["DAPI_2", "DAPI_4_trunc_0", "DAPI_4_trunc"]
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
image_layer = viewer.add_image(DAPI_4, scale=scale, colormap='magenta', blending='additive',visible=False)
#image_layer = viewer.add_image(betaCatenin, scale=scale, colormap='inferno', blending='additive',visible=False)
#image_layer = viewer.add_image(MapK, scale=scale, colormap='inferno', blending='additive',visible=False)
#image_layer = viewer.add_image(pSmad23, scale=scale, colormap='inferno', blending='additive',visible=False)
#image_layer = viewer.add_image(pSmad15, scale=scale, colormap='inferno', blending='additive',visible=False)
#image_layer = viewer.add_image(DAPI_4_shift, scale=scale, colormap='green', blending='additive',visible=False)
#image_layer = viewer.add_image(DAPI_4_trunc_0, scale=scale, colormap='red', blending='additive',visible=False)
#image_layer = viewer.add_image(DAPI_4_trunc, scale=scale, colormap='red', blending='additive',visible=False)
#image_layer = viewer.add_image(pSmad15_shift, scale=scale, colormap='inferno', blending='additive',visible=False)
#image_layer = viewer.add_labels(Nucleiseg, scale=scale, blending='additive',visible=False)
#image_layer = viewer.add_labels(DAPI_2_bool, scale=scale, blending='additive',visible=False)
#image_layer = viewer.add_labels(cell_seg, scale=scale, blending='additive',visible=False)
#image_layer = viewer.add_image(sphere, scale=scale, blending='additive',visible=False)









