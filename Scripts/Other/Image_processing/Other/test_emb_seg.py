#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 09:09:04 2023

@author: floriancurvaia
"""
import numpy as np

from abbott.h5_files import *

import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

import pandas as pd

fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B08_px+1076_py-0189.h5'

#print(h5_summary(fn))

f=h5py.File(fn)
#print(f.keys())

A=np.array(f["ch_02"]["1"])
B=np.array(f["lbl_nuc_raw2"])
C=np.array(f["ch_00"]["1"])
D=np.array(f["lbl_embryo"])
E=np.array(f["lbl_cells"])

fn1="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/B08_px+1076_py-0189.csv"

features=pd.read_csv(fn1, sep=";", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "structure", "pSmad2/3_2_Mean" ]) #
features[["Centroid_x","Centroid_y", "Centroid_z"]]=features[["Centroid_x","Centroid_y", "Centroid_z"]]/[0.65,0.65,1]
feat_filt=features[(features.structure=='cells')]
points=np.zeros(A.shape)
all_idx=np.array(feat_filt[["Centroid_z","Centroid_y", "Centroid_x"]]).T.astype(int)
points[all_idx[0], all_idx[1], all_idx[2]]=1
r = 10

xx, yy = np.mgrid[-r:r+1, -r:r+1]
circle = xx**2 + yy**2 <= r**2

for point in points:
    condition = np.where(point==1) # np.where(arr > .5) to benchmark 60k circles
    for x,y in zip(*condition):
        # valid indices of the array
        i = slice(max(x-r,0), min(x+r+1, point.shape[1]))
        j = slice(max(y-r,0), min(y+r+1, point.shape[0]))
    
        # visible slice of the circle
        ci = slice(abs(min(x-r, 0)), circle.shape[1] - abs(min(point.shape[1]-(x+r+1), 0)))
        cj = slice(abs(min(y-r, 0)), circle.shape[0] - abs(min(point.shape[0]-(y+r+1), 0)))
        
        point[j, i] += circle[cj, ci]

#A=A/np.max(A)
#plt.show()
#plt.plot(f["ch_03"])


fig1=plt.figure(1)
plt.clf()
idx0 = 3
n = plt.imshow(D[idx0], vmin=0, vmax=1) #, cmap="grey"
q = plt.imshow(E[idx0], vmin=0, vmax=0.75)
k = plt.imshow(C[idx0], vmin=0, vmax=100, alpha=0.65)
l = plt.imshow(A[idx0], vmin=np.min(A), vmax=np.max(A)*0.8, alpha=0.25)
m = plt.imshow(B[idx0], vmin=0, vmax=1, alpha=0.15)
p = plt.imshow(points[idx0], vmin=0, vmax=1, alpha=0.15)

channels=[n, q, k, l, m, p]

#l = plt.imshow(A[idx0])
axidx = plt.axes([0.25, 0, 0.65, 0.03])
slidx = Slider(axidx, 'z-stack', 0, A.shape[0]-1, valinit=idx0, valfmt='%d')

rax = plt.axes([0.05, 0.4, 0.1, 0.15])
labels = ["emb_seg", "cell_seg", "bcat", "DAPI", "nuc_seg", "centroids"]
visibility = [chan.get_visible() for chan in channels]
check = CheckButtons(rax, labels, visibility)


def func(label):
    index = labels.index(label)
    channels[index].set_visible(not channels[index].get_visible())
    plt.draw()

check.on_clicked(func)

def update(val):
    idx = slidx.val
    l.set_data(A[int(idx)])
    m.set_data(B[int(idx)])
    k.set_data(C[int(idx)])
    n.set_data(D[int(idx)])
    q.set_data(E[int(idx)])
    p.set_data(points[int(idx)])
    fig1.canvas.draw_idle()
slidx.on_changed(update)

plt.show()



