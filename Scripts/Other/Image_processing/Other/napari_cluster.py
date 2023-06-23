#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:27:09 2023

@author: floriancurvaia
"""

import h5py

import napari

from abbott.h5_files import *

import numpy as np

import argparse

CLI=argparse.ArgumentParser()
CLI.add_argument(
  "im",  # name on the CLI - drop the `--` for positional/required parameters
  type=str
)

args = CLI.parse_args()

#im="B08_px+1076_py-0189" #B07_px+1257_py-0474 B08_px+1076_py-0189 B03_px-0545_py-1946 B05_px+1522_py-1087 B06_px-0493_py+0146

im=args.im


fn="/data/active/sshami/20220716_experiment3_aligned/"+im+".h5"

#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Compressed_files_aligned/"+im+".h5"

#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Compressed_files_pre/"+im+".h5"

#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Compressed_files/"+im+".h5"

#fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/'+im+".h5"

#fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Compressed_files/'+im+".h5"

#path_in_sph= "/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/Arrays/out_sphere_"#"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/"#"Spheres_fit/"

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
"""
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
"""
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'DAPI', 'level': 1, 'cycle': cycle}):
        DAPI_0 = to_numpy(dset)
"""
cycle=1
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'DAPI', 'level': 1, 'cycle': cycle}):
        cycle = dset.attrs['cycle']
        scale = dset.attrs['element_size_um']
        DAPI_1 = to_numpy(dset)
        
"""
cycle=2
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'MapK', 'level': 1, 'cycle': cycle}):
        MapK = to_numpy(dset)
        
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'pSmad2/3', 'level': 1, 'cycle': cycle}):
        pSmad23 = to_numpy(dset)

with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'DAPI', 'level': 1, 'cycle': cycle}):
        DAPI_2 = to_numpy(dset)
        
cycle=3
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'DAPI', 'level': 1, 'cycle': cycle}):
        DAPI_3 = to_numpy(dset)
"""        
cycle=4
 
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'pSmad1/5', 'level': 1, 'cycle': cycle}):
        pSmad15 = to_numpy(dset)
"""
with h5py.File(fn) as f:
    for dset in h5_select(f, {'stain': 'DAPI', 'level': 1, 'cycle': cycle}):
        DAPI_4 = to_numpy(dset)
"""

#sphere=np.load(path_in_sph+im+".npy") #(path_in_sph+"out_sphere_"+im+".npy")

viewer = napari.Viewer()

#Dist=np.load("min_dists_y_x.npy")
#image_layer=viewer.add_points(data=coords, ndim=3, properties=Dist)
#image_layer = viewer.add_image(DAPI_0, scale=scale, colormap='magenta', blending='additive',visible=False)
image_layer = viewer.add_image(DAPI_1, scale=scale, colormap='magenta', blending='additive',visible=False)
#image_layer = viewer.add_image(DAPI_2, scale=scale, colormap='magenta', blending='additive',visible=False)
#image_layer = viewer.add_image(DAPI_3, scale=scale, colormap='magenta', blending='additive',visible=False)
#image_layer = viewer.add_image(DAPI_4, scale=scale, colormap='magenta', blending='additive',visible=False)
image_layer = viewer.add_image(betaCatenin, scale=scale, colormap='inferno', blending='additive',visible=False)
#image_layer = viewer.add_image(MapK, scale=scale, colormap='inferno', blending='additive',visible=False)
#image_layer = viewer.add_image(pSmad23, scale=scale, colormap='inferno', blending='additive',visible=False)
image_layer = viewer.add_image(pSmad15, scale=scale, colormap='inferno', blending='additive',visible=False)
#image_layer = viewer.add_image(betaCateninseg, scale=scale, colormap='inferno', blending='additive',visible=False)
#image_layer = viewer.add_labels(Nucleiseg, scale=scale, blending='additive',visible=False)
#image_layer = viewer.add_labels(emb_seg, scale=scale, blending='additive',visible=False)
image_layer = viewer.add_labels(cell_seg, scale=scale, blending='additive',visible=False)
#image_layer = viewer.add_image(sphere, scale=scale, blending='additive',visible=False)









