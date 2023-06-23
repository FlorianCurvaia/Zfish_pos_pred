#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:30:10 2023

@author: floriancurvaia
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import scipy.stats as scst

import plotly.express as px

im="B07_px+1257_py-0474"
fn_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv" #B08_px+1076_py-0189_w_dist_sph_simp.csv"
feat_filt=pd.read_csv(fn_dist, sep=",", index_col=False)

#plt.scatter(feat_filt.PhysicalSize, feat_filt.dist_out, c=feat_filt.KMeans_3)


scaler = StandardScaler()
#init_centers=np.array([[13620, 62.5], [13010, 8.9], [3100, 31]])
#init_centers=np.array([[12541.60805147059, 62.01264938445141], [14267.907663043474, 10.911887499867083], [3158.7920548301367, 29.126187516429678]])
#scaled_features = scaler.fit_transform(np.array([scst.zscore(np.log(feat_filt.NucSize)), scst.zscore(feat_filt.PhysicalSize)]).T) #feat_filt.PhysicalSize.to_numpy().reshape(-1,1)
scaled_features = scaler.fit_transform(np.array([feat_filt.NucSize, feat_filt.PhysicalSize]).T)

#scaled_features = scaler.fit_transform(np.array([feat_filt.PhysicalSize,feat_filt.dist_out, feat_filt.aspect_ratio_max]).T)
c=np.mean(scaled_features[np.where((feat_filt.NucSize.to_numpy()>1500)& (feat_filt.PhysicalSize.to_numpy()<20000))], axis=0) #(feat_filt.NucSize.to_numpy()>1500)&

#print(np.mean(scaled_features[np.where(feat_filt.PhysicalSize.to_numpy()>13000)], axis=0))

a=np.mean(scaled_features[np.where((feat_filt.NucSize.to_numpy()>1000) & (feat_filt.PhysicalSize.to_numpy()>20000))], axis=0)

b=np.mean(scaled_features[np.where((feat_filt.NucSize.to_numpy()>3000) & (feat_filt.PhysicalSize.to_numpy()<15000))], axis=0)
init_means=np.array([a, b, c])

kmeans_3= KMeans(n_clusters=3, random_state=42, init=np.array([a, b, c]))
#kmeans_2= KMeans(n_clusters=2, random_state=42)
#kmeans_2.fit(scaled_features)
kmeans_3.fit(scaled_features)
#feat_filt["KMeans_2_c"]=kmeans_2.labels_
feat_filt["KMeans_3_c"]=kmeans_3.labels_

dbscan=DBSCAN(eps=0.5)
dbscan.fit(scaled_features)
feat_filt["DBSCAN_05"]=dbscan.labels_




gmm = GaussianMixture(n_components = 3, covariance_type="spherical",  means_init=np.array([a, b,  c])) #, weights_init=[ 0.998, 0.002,]
gmm.fit(scaled_features)
labels = gmm.predict(scaled_features)
feat_filt["GMM"]=labels


path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Clusters/"

#print(np.mean(feat_filt.loc[(feat_filt.PhysicalSize>10000) & (feat_filt.dist_out>33)]).PhysicalSize)

#print(np.mean(feat_filt.loc[(feat_filt.PhysicalSize>10000) & (feat_filt.dist_out>33)]).dist_out)
#print(np.mean(feat_filt.loc[(feat_filt.PhysicalSize>10000) & (feat_filt.dist_out<33)]).PhysicalSize)
#print(np.mean(feat_filt.loc[(feat_filt.PhysicalSize>10000) & (feat_filt.dist_out<33)]).dist_out)


plt.figure(1)
plt.clf()
plt.scatter(feat_filt.NucSize, feat_filt.PhysicalSize, c=feat_filt.KMeans_3_c) #, s=(feat_filt.KMeans_3_c+1)*100, c=feat_filt.betaCatenin_cyto)

plt.figure(2)
plt.clf()
plt.scatter(feat_filt.NucSize, feat_filt.PhysicalSize, c=feat_filt.DBSCAN_05) #, s=(feat_filt.DBSCAN_05+1)*100, c=feat_filt.betaCatenin_cyto)

plt.figure(3)
plt.clf()
plt.scatter(feat_filt.NucSize, feat_filt.PhysicalSize, c=feat_filt.GMM) #, s=(feat_filt.DBSCAN_05+1)*100, c=feat_filt.betaCatenin_cyto)
plt.savefig(path_out_im+"Size_Dist_scatter_GMM_"+im)


plt.figure(4)
plt.clf()
plt.scatter(scaled_features[:,0], scaled_features[:,1], c=feat_filt.KMeans_3_c) #, s=(feat_filt.KMeans_3_c+1)*100, c=feat_filt.betaCatenin_cyto)

plt.figure(5)
plt.clf()
plt.scatter(feat_filt.NucSize, feat_filt.aspect_ratio_max, c=feat_filt.DBSCAN_05) #, s=(feat_filt.DBSCAN_05+1)*100, c=feat_filt.betaCatenin_cyto)

plt.figure(6)
plt.clf()
plt.scatter(feat_filt.NucSize, feat_filt.aspect_ratio_max, c=feat_filt.GMM)




fig7 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="GMM", opacity=1, color_continuous_scale="turbo")
fig7.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig8.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig7.write_html(path_out_im+"GMM_"+im+".html")

fig8 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="KMeans_3_c", opacity=1, color_continuous_scale="turbo")
fig8.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig8.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig8.write_html(path_out_im+"KMeans_"+im+".html")

fig9 = px.scatter_3d(feat_filt, x='x_corr', y='y_corr', z='z_corr', color="DBSCAN_05", opacity=1, color_continuous_scale="turbo")
fig9.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig8.write_html(path_out_im+"dist_out_simp_"+im+"_"+str(n_bins)+".html")
fig9.write_html(path_out_im+"DBSCAN_"+im+".html")





