#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:51:50 2023

@author: floriancurvaia
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

#f =h5py.File('/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/To_shayan/20220716_experiment3_aligned/B08_px+1076_py-0189.h5', "r")
f=h5py.File('/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/results/B08_px+1076_py-0189.h5')
print(f.keys())

A=np.array(f["ch_02"]["1"])
B=np.array(f["lbl_nuc_raw2"])
C=np.array(f["ch_00"]["1"])
#A=A/np.max(A)
#plt.show()
#plt.plot(f["ch_03"])


fig1=plt.figure(1)
plt.clf()
idx0 = 3
k = plt.imshow(C[idx0], vmin=0, vmax=100)
l = plt.imshow(A[idx0], vmin=np.min(A), vmax=np.max(A)*0.8, alpha=0.60)
m = plt.imshow(B[idx0], vmin=0, vmax=1, alpha=0.25)

#l = plt.imshow(A[idx0])
axidx = plt.axes([0.25, 0, 0.65, 0.03])
slidx = Slider(axidx, 'z-stack', 0, A.shape[0]-1, valinit=idx0, valfmt='%d')

def update(val):
    idx = slidx.val
    l.set_data(A[int(idx)])
    m.set_data(B[int(idx)])
    k.set_data(C[int(idx)])
    fig1.canvas.draw_idle()
slidx.on_changed(update)

plt.show()

"""
fig2=plt.figure(2)
plt.clf()

#k = plt.imshow(C[idx0], vmin=0, vmax=100)
#l = plt.imshow(A[idx0], vmin=np.min(A), vmax=np.max(A), alpha=1)
m = plt.imshow(B[idx0], vmin=0, vmax=1, alpha=1)

#l = plt.imshow(A[idx0])
axidx = plt.axes([0.25, 0, 0.65, 0.03])
slidx1 = Slider(axidx, 'z-stack', 0, A.shape[0]-1, valinit=idx0, valfmt='%d')

def update_1(val):
    idx = slidx1.val
#    l.set_data(A[int(idx)])
    m.set_data(B[int(idx)])
#    k.set_data(C[int(idx)])
    fig2.canvas.draw_idle()
slidx1.on_changed(update_1)

plt.show()

fig3=plt.figure(3)
plt.clf()

#k = plt.imshow(C[idx0], vmin=0, vmax=100)
l = plt.imshow(A[idx0], vmin=np.min(A), vmax=np.max(A), alpha=1)
m = plt.imshow(B[idx0], vmin=0, vmax=1, alpha=0.25)

#l = plt.imshow(A[idx0])
axidx = plt.axes([0.25, 0, 0.65, 0.03])
slidx2 = Slider(axidx, 'z-stack', 0, A.shape[0]-1, valinit=idx0, valfmt='%d')

def update_2(val):
    idx = slidx2.val
    l.set_data(A[int(idx)])
    m.set_data(B[int(idx)])
#    k.set_data(C[int(idx)])
    fig3.canvas.draw_idle()
slidx2.on_changed(update_2)

plt.show()
"""