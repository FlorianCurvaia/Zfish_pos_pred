#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:46:01 2023

@author: floriancurvaia
"""


import numpy as np

import time

import pandas as pd


import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt

start_time_0=time.time()
n_bins=72
im="B03_px-0545_py-1946" #B04_px+0114_py-1436, B08_px+1076_py-0189, B07_px+1257_py-0474, B03_px-0545_py-1946, B02_px-0280_py+1419
#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/B08_px+1076_py-0189.csv"
fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_edge_corr_mi.csv" #fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_w_dist_edge_corr_elli.csv"
#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/features/B02_px-0280_py+1419.csv"
out_ed_samp=pd.read_csv("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_out_ed_samp.csv") #out_ed_samp=pd.read_csv("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_out_ed_samp_elli.csv")
features=pd.read_csv(fn, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "structure", "pSmad2/3_2_Mean", "theta","phi","d_corr", "phi_bin", "theta_bin", "d_i", "d_o", "dist_out", "dist_in", "rank_out", "d_sor_r", "dist_rel_in", "dist_rel_out", "dist_rel_out_r", "dist_rel_in_r", "d_sor_r_2", "rank_out_r"])

print("Read and load file --- %s seconds ---\n" % (time.time() - start_time_0))

start_time_1=time.time()
feat_filt=features[(features.structure=='cells')]
origin=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/out_sphere_origin_"+im+".npy") #origin=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/out_elli_origin_"+im+".npy")
out_sphere=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/out_sphere_"+im+".npy")
in_sphere=np.load("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Spheres_fit/in_sphere_"+im+".npy")

out_sphere=np.array(np.where(out_sphere==True))
in_sphere=np.array(np.where(in_sphere==True))

out_df=pd.DataFrame(out_sphere.T, columns=["Z_out", "Y_out", "X_out"])
out_df[["Z_out", "Y_out", "X_out"]]=out_df[["Z_out", "Y_out", "X_out"]]-[origin[2],origin[1],origin[0]]
out_df[["Z_out", "Y_out", "X_out"]]=out_df[["Z_out", "Y_out", "X_out"]]*[1, 0.65, 0.65]


in_df=pd.DataFrame(in_sphere.T, columns=["Z_in", "Y_in", "X_in"])
in_df[["Z_in", "Y_in", "X_in"]]=in_df[["Z_in", "Y_in", "X_in"]]-[origin[2],origin[1],origin[0]]
in_df[["Z_in", "Y_in", "X_in"]]=in_df[["Z_in", "Y_in", "X_in"]]*[1, 0.65, 0.65]



features[["Centroid_x","Centroid_y", "Centroid_z"]]=features[["Centroid_x","Centroid_y", "Centroid_z"]]-[origin[0],origin[1],origin[2]]

del out_sphere, in_sphere

def appendSpherical_pd_cm(df): #from https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    xy = df["Centroid_x"]**2+df["Centroid_y"]**2
    df["r"] = np.sqrt(xy + df["Centroid_z"]**2)
    #df["theta"] = pi+np.arctan2(np.sqrt(xy), df["Centroid_z"])
    df["theta"]= np.arccos(df["Centroid_z"]/df["r"])
    #df["phi"] = np.arctan2(df["Centroid_y"], df["Centroid_x"])
    df["phi"] = np.sign(df["Centroid_y"])*np.arccos(df["Centroid_x"]/np.sqrt(xy))
    return df

#features=appendSpherical_pd_cm(features)

print("Generate spherical coordinates --- %s seconds ---\n" % (time.time() - start_time_1))

path_out="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/D_in_microns/"

feat_filt=features[(features.structure=='cells')]
del features
feat_filt["d_r"]=feat_filt.dist_out/(feat_filt.dist_out+feat_filt.dist_in)
feat_filt["diff_d_o"]=np.abs(feat_filt.dist_out-feat_filt.d_o)
feat_filt["diff_d_o"]=feat_filt["diff_d_o"]/np.max(feat_filt["diff_d_o"])

fig0 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color='pSmad2/3_2_Mean', opacity=1)
fig0.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig0.write_html(path_out+"pSmad23_2_Mean_"+im+"_"+str(n_bins)+".html") #Modifiy the html file
#fig0.show()

fig1 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color='theta', opacity=1)
fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig1.write_html(path_out+"theta_"+im+"_"+str(n_bins)+".html") #Modifiy the html file
#fig1.show()

fig2 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color='phi', opacity=1)
fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig2.write_html(path_out+"phi_"+im+"_"+str(n_bins)+".html") #Modifiy the html file
#fig2.show()

fig3 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="d_o", opacity=1)
fig3.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig3.write_html(path_out+"d_o_"+im+"_"+str(n_bins)+".html") #Modifiy the html file
#fig3.show()

fig4 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="d_i", opacity=1)
fig4.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig4.write_html(path_out+"d_i_"+im+"_"+str(n_bins)+".html") #Modifiy the html file

fig5 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="d_corr", opacity=1)
fig5.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig5.write_html(path_out+"d_corr_"+im+"_"+str(n_bins)+".html")

fig6 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="d_r", opacity=1)
fig6.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig6.write_html(path_out+"d_r_"+im+"_"+str(n_bins)+".html")

fig7 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="diff_d_o", opacity=1)
fig7.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig7.write_html(path_out+"diff_d_o_"+im+"_"+str(n_bins)+".html")

fig8 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="dist_out", opacity=1)
fig8.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig8.write_html(path_out+"dist_out_"+im+"_"+str(n_bins)+".html")

fig12 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="dist_rel_out", opacity=1)
fig12.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig12.write_html(path_out+"dist_rel_out_"+im+"_"+str(n_bins)+".html")

fig13 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="rank_out", opacity=1)
fig13.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig13.write_html(path_out+"rank_out_"+im+"_"+str(n_bins)+".html")

fig14 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="d_sor_r", opacity=1)
fig14.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig14.write_html(path_out+"d_sor_r_"+im+"_"+str(n_bins)+".html")

fig15 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="d_sor_r_2", opacity=1)
fig15.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig15.write_html(path_out+"d_sor_r_2_"+im+"_"+str(n_bins)+"_elli"+".html")

fig16 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="dist_in", opacity=1)
fig16.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig16.write_html(path_out+"dist_in_"+im+"_"+str(n_bins)+".html")

fig17 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="rank_out_r", opacity=1)
fig17.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig17.write_html(path_out+"rank_out_r_"+im+"_"+str(n_bins)+".html")

fig18 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="dist_rel_out_r", opacity=1)
fig18.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig18.write_html(path_out+"dist_rel_out_r_"+im+"_"+str(n_bins)+".html")

fig19 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="dist_rel_in_r", opacity=1)
fig19.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig19.write_html(path_out+"dist_rel_in_r_"+im+"_"+str(n_bins)+".html") #fig19.write_html(path_out+"dist_rel_in_r_"+im+"_"+str(n_bins)+"_elli"+".html")


"""
fig22 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="dist_rel_out", opacity=1)
px.scatter_3d(in_df, x="X_in", y="Y_in", z="Z_in", opacity=0.5)
fig22.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig22.write_html(path_out+"dist_rel_out_w_in_sphere_"+im+"_"+str(n_bins)+".html")
"""
#fig23 = px.scatter_3d(feat_filt, x='Centroid_x', y='Centroid_y', z='Centroid_z', color="dist_rel_out", opacity=1)
#fig23.add_scatter_3d(out_df, x="X_out", y="Y_out", z="Z_out", opacity=0.5)
#fig23.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#fig23.write_html(path_out+"dist_rel_out_w_out_sphere_"+im+"_"+str(n_bins)+".html")

#fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
#fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
"""
fig23 = go.Figure(data=[
    go.Surface(z=out_df.Z_out.to_numpy(), x=out_df.X_out.to_numpy(), y=out_df.Y_out.to_numpy(), opacity=0.8 ),
    go.Scatter3d(x=feat_filt.Centroid_x.to_numpy(), y=feat_filt.Centroid_y.to_numpy(), z=feat_filt.Centroid_z.to_numpy(), mode='markers', marker=dict(
        size=2,
        color=feat_filt.dist_rel_out.to_numpy(),                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    ))
])
fig23.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig23.write_html(path_out+"dist_rel_out_w_out_sphere_"+im+"_"+str(n_bins)+".html")
"""

top_emb=feat_filt.loc[feat_filt.Centroid_z<-450]

fig9=plt.figure(9)
plt.clf()
plt.hist(top_emb.d_corr, bins=25, density=True)
plt.hist(feat_filt.d_corr, bins=25, density=True, alpha=0.5)

fig10=plt.figure(10)
plt.clf()
plt.hist(top_emb.d_r, bins=25, density=True)
plt.hist(feat_filt.d_r, bins=25, density=True, alpha=0.5)

fig11=plt.figure(11)
plt.clf()
plt.hist(top_emb.diff_d_o, bins=25, density=True)
plt.hist(feat_filt.diff_d_o, bins=25, density=True, alpha=0.5)

fig20=plt.figure(20)
plt.clf()
plt.hist(top_emb.dist_rel_out, bins=25, density=True)
plt.hist(feat_filt.dist_rel_out, bins=25, density=True, alpha=0.5)

fig21=plt.figure(21)
plt.clf()
plt.hist(top_emb.dist_rel_out_r, bins=25, density=True)
plt.hist(feat_filt.dist_rel_out_r, bins=25, density=True, alpha=0.5)

#feat_filt.to_csv("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"+im+"_sphr_dist.csv", sep=",")











