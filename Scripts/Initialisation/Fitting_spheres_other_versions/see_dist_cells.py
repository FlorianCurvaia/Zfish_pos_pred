#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:36:32 2023

@author: floriancurvaia
"""

import numpy as np

import h5py
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

#B04_px+0114_py-1436, B08_px+1076_py-0189, B07_px+1257_py-0474, B03_px-0545_py-1946, B02_px-0280_py+1419, C05_px+0198_py+1683

fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B08_px+1076_py-0189.h5"


with h5py.File(fn) as f:
    dist_cells=np.array(f["dist_sph"])


fig1=plt.figure(1)
plt.clf()
idx0 = 3
m = plt.imshow(dist_cells[idx0], vmin=0, vmax=1)



axidx = plt.axes([0.25, 0, 0.65, 0.03])
slidx = Slider(axidx, 'z-stack', 0, dist_cells.shape[0]-1, valinit=idx0, valfmt='%d')

def update(val):
    idx = slidx.val
    m.set_data(dist_cells[int(idx)])
    fig1.canvas.draw_idle()
slidx.on_changed(update)

plt.show()