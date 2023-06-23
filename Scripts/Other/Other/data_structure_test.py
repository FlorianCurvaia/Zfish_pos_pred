#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:33:56 2023

@author: floriancurvaia
"""
import numpy as np
from skimage.io import imread
import time

#import os
from abbott.conversions import to_numpy
#import napari

start_time_0=time.time()

image=imread("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/C07_px+1929_py-0176-1.tif")
print("Reading image --- %s seconds ---\n" % (time.time() - start_time_0))
#image1=np.swapaxes(image,0,1)

#start_time_1=time.time()

#image2=to_numpy("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/C07_px+1929_py-0176-1.tif")
#print("Reading image --- %s seconds ---\n" % (time.time() - start_time_1))

#for i in image1:
#    print(i.shape)

class embryo:
    """
    A class to handle embryos as 4-Dimensional np arrays.
    """
    def __init__(self, image, channels_labels):
        
        self.labs=dict(zip(list(channels_labels), list(range(len(channels_labels))))) #Storing the labels with their index in a dictionnary 
        
        self.array=np.swapaxes(image,0,1) #Putting the different channels as the first dimension, so each embryo can be entirely accessed for a given channel using dimension 2, 3 and 4
        
        
    def get_channel(self, chan_name):
        """
        

        Parameters
        ----------
        chan_name : str
            the name of the chanel for which the image of the embryo should be returned.

        Returns
        -------
        3D np array
            3-D array corresponding image of the embryo for the desired channel.

        """
        index= self.labs[chan_name]
        return self.array[index]
        


def appendSpherical_np(xyz): # from https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

chans=["B-cat", "pERM", "pMyosin", "DAPI", "pERK", "pSmad2/3", "pSmad1/5", "p56"]
start_time_2=time.time()
emb1=embryo(image, chans)
print("Creating embryo object --- %s seconds ---\n" % (time.time() - start_time_2))
del image
#del image1

#viewer = napari.Viewer()
#new_layer = viewer.add_image(emb1.get_channel("B-cat"))
#new_layer1 = viewer.add_image(np.swapaxes(emb1.get_channel("B-cat"), 0, 2))
#napari.run()
print("Whole script --- %s seconds ---\n" % (time.time() - start_time_0))