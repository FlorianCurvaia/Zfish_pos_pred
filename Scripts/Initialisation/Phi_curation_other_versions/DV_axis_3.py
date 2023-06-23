#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:50:15 2023

@author: floriancurvaia
"""



import numpy as np

import time

import pandas as pd

import math

import plotly.express as px

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
import scipy.stats as stats

plt.ioff()

margin=0.8
stains=["betaCatenin_nuc", "pSmad1/5_nc_ratio"] #"pSmad1/5_nc_ratio"
start_time_0=time.time()
#results=pd.DataFrame(columns=["Emb", "hpf","min_2", "min_2_diff", "min_r2", "min_x2", "min_phi", "min_phi_diff", "max_r2",  "max_x2", "max_phi", "max_phi_diff"])
#results=pd.DataFrame(columns=["Emb", "hpf", "min_phi", "min_phi_diff"])
results=pd.DataFrame(columns=["Emb", "hpf","min_2", "min_2_diff", "min_r2", "min_x2", "min_phi", "min_phi_diff"])
#path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/DV_axis/Opt/"
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/DV_axis/"
#path_out_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/"
path_out_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
path_out_dist="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/"
images=['D06_px-0377_py-1358', 'D06_px+1741_py-0131', 'D06_px+0217_py+1735', 'D06_px-1055_py-0118', 
        'B07_px-0202_py+0631', 'B07_px+1257_py-0474', 'B07_px+0030_py-1959', 'B07_px-2056_py-0041', 
        'B07_px+0282_py+1729', 'C07_px+1425_py+0902', 'C07_px+1929_py-0176', 'C07_px-0784_py-0616', 
        'C07_px+0243_py-1998', 'C07_px+0301_py+1212', 'D07_px-0500_py+0670', 'D07_px+0999_py-1281']

images_1=["B02_px-0280_py+1419", "B03_px-0545_py-1946", "C04_px-0816_py-1668", "B05_px+1522_py-1087", "C05_px+0198_py+1683", 
        "D05_px-0738_py-1248", "B06_px-0493_py+0146", "D06_px-1055_py-0118", "B07_px+1257_py-0474", "B08_px+1076_py-0189"]
images+=images_1

to_remove=['D06_px+1741_py-0131', 'D06_px-1055_py-0118', 'B07_px+1257_py-0474', 'C07_px+1929_py-0176', "B07_px-0202_py+0631", "C07_px+0243_py-1998"]

images.remove('B07_px-0202_py+0631')
images.remove("C07_px+0243_py-1998")
#images=list(set(images)-set(to_remove))

time_emb={"B02":3.3, "C02":3.3, "D02":3.3, "B03":3.7, "C03":3.7, "D03":3.7, "B04":4.3, "C04":4.3, "D04":4.3,
          "B05":4.7, "C05":5.3, "D05":5.7, "B06":5.7, "C06":5.7, "D06":6.0, "B07":6.0, "C07":6.0, "D07":6.0,
          "B08":7.0, "C08":7.0, "D08":7.0}


margin=0.8

colnames=['Embryo', 'Cell ID1', 'Cell ID2', 'Cell ID3', 'Cell ID4', 'Cell ID5', 'Cell ID6']
dorsal_cells=pd.read_csv("/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/CellIDS.csv", sep=";", index_col=False, usecols=colnames)
curated_ids_emb={}
images=dorsal_cells.Embryo.tolist()
for im in images:
    dors_ids=dorsal_cells.loc[dorsal_cells.Embryo==im, dorsal_cells.columns != 'Embryo'].values.tolist()[0]
    if dorsal_cells.loc[dorsal_cells.Embryo==im].isnull().values.any():
        to_remove.append(im)
    else:
        curated_ids_emb[im]=dors_ids
images=list(set(images) - set(to_remove))
            
for idx in range(len(images)):
    im=images[idx]
    row_to_fill=[]
    fn_dist=path_out_dist+im+"_w_dist_sph_simp.csv" #B08_px+1076_py-0189_w_dist_sph_simp.csv"
    feat_filt=pd.read_csv(fn_dist, sep=",", index_col=False)
    n_bins=36
    shifts=np.linspace(-math.pi, math.pi, 60)
    #print("load files --- %s seconds ---\n" % (time.time() - start_time_0))
    curated_ids=curated_ids_emb[im]
    
    start_time_1=time.time()
    well=im.split("_")[0]
    row_to_fill.append(im)
    row_to_fill.append(time_emb[well])
    #curated_ids=curated_ids_emb[im]
    mean_phi_cur=feat_filt.loc[feat_filt.Label.isin(curated_ids)]["phi"].mean()
    mean_pSmad_cur=feat_filt.loc[feat_filt.Label.isin(curated_ids)]["pSmad1/5_nc_ratio"].mean() #"pSmad1/5_nc_ratio"
    cur_cells=feat_filt.loc[feat_filt.Label.isin(curated_ids)]

    #print(mean_pSmad_cur)
    cur_phi_0=mean_phi_cur

    
    #print("find curated phi mean --- %s seconds ---\n" % (time.time() - start_time_1))
    for stain in stains:
        if stain=="pSmad1/5_nc_ratio":
            y_min=feat_filt[stain].quantile(0.01)
            y_max=feat_filt[stain].quantile(0.99)
            feat_filt=feat_filt.loc[feat_filt[stain]<y_max]
            #feat_filt=feat_filt.loc[feat_filt[stain]>y_min]
        feat_filt[stain]=stats.zscore(feat_filt[stain])
    #feat_filt=feat_filt.loc[feat_filt[stain]>1]
    feat_filt["rel_theta"]=feat_filt.theta/np.max(feat_filt.theta)
    min_bins={}
    min_bins_b={}
    #min_bins_diff={}
    #min_bins_2={}
    start_time_2=time.time()
    #if stain == "betaCatenin_nuc":
    #    margin_loc=feat_filt.rel_theta.quantile(margin)
    #    feat_filt=feat_filt.loc[feat_filt.rel_theta>margin_loc].copy()
    for shift in shifts:
    
    
        new_phi_sign=np.sign(shift)
    
        if new_phi_sign==-1:
            feat_filt["phi_temp"]=np.abs(shift)+feat_filt["phi"]
            feat_filt.loc[(feat_filt["phi_temp"]>math.pi), "phi_temp"]=feat_filt["phi_temp"]-2*math.pi
        elif new_phi_sign==1:
            feat_filt["phi_temp"]=feat_filt["phi"]-shift
            feat_filt.loc[(feat_filt["phi_temp"]<-1*math.pi), "phi_temp"]=feat_filt["phi_temp"]+2*math.pi
        else:
            feat_filt["phi_temp"]=feat_filt["phi"]
            
        
        feat_filt["phi_temp"]=np.abs(feat_filt["phi_temp"])
        phi_bins=np.linspace(0, math.pi, n_bins)
        #phi_bins=np.linspace(-math.pi, math.pi, n_bins)
        labels = range(1, n_bins)
        theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)
        
        feat_filt['phi_bin_DV'] = pd.to_numeric(pd.cut(x = feat_filt['phi_temp'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
        
        margin_loc=feat_filt.rel_theta.quantile(margin)
        feat_filt_marg=feat_filt.loc[feat_filt.rel_theta>margin_loc].copy()
        
        means_psmad15_phi=feat_filt.groupby(['phi_bin_DV'])["pSmad1/5_nc_ratio"].mean() #"pSmad1/5_nc_ratio" #"betaCatenin_nuc"
        means_bcat_phi=feat_filt_marg.groupby(['phi_bin_DV'])["betaCatenin_nuc"].mean()
        

        
        feat_filt["pSmad1/5_nc_ratio_mean_phi"]=0
        feat_filt_marg["betaCatenin_nuc_mean_phi"]=0
        
        for i in range(1, n_bins):
            if i in means_psmad15_phi.index:
                feat_filt.loc[(feat_filt["phi_bin_DV"]==i), "pSmad1/5_nc_ratio_mean_phi"]=means_psmad15_phi.loc[(i)]#math.log(means_psmad15_phi.loc[(i)])
            if i in means_bcat_phi.index:
                feat_filt_marg.loc[(feat_filt_marg["phi_bin_DV"]==i), "betaCatenin_nuc_mean_phi"]=means_bcat_phi.loc[(i)]
                    
        #bin_min=means_psmad15_phi.loc[means_psmad15_phi==min(means_psmad15_phi)].index[0]
        #bin_range=[phi_bins[bin_min-1], phi_bins[bin_min]]
        new_phi_0_p=feat_filt.loc[feat_filt["pSmad1/5_nc_ratio_mean_phi"]==min(means_psmad15_phi), "phi"].mean()#np.mean(bin_range)
        #new_phi_0=feat_filt.loc[feat_filt["phi_bin_DV"]==1, "phi"].mean()#np.mean(bin_range)
        
        #min_bins[new_phi_0]=means_psmad15_phi.loc[(1)]
        #min_bins_diff[new_phi_0]=new_phi_0-cur_phi_0
        #min_bins_2[new_phi_0]=bin_min
        
        #bin_min_b=means_bcat_phi.loc[means_bcat_phi==min(means_bcat_phi)].index[0]
        #bin_range=[phi_bins[bin_min-1], phi_bins[bin_min]]
        new_phi_0_b=feat_filt_marg.loc[feat_filt_marg["betaCatenin_nuc_mean_phi"]==min(means_bcat_phi), "phi"].mean()#np.mean(bin_range)
        #new_phi_0=feat_filt.loc[feat_filt["phi_bin_DV"]==1, "phi"].mean()#np.mean(bin_range)
        #new_phi_0=(new_phi_0_p+new_phi_0_b)/2
        #min_bins[new_phi_0]=(min(means_psmad15_phi)+min(means_bcat_phi))/2
        #min_bins[new_phi_0_b]=min(means_bcat_phi)
        min_bins[new_phi_0_p]=min(means_psmad15_phi)
        
        
        """
        new_phi_0=min(new_phi_0_p, new_phi_0_b)
        if new_phi_0==new_phi_0_p:
            min_bins[new_phi_0_p]=min(means_psmad15_phi)
        else:
            min_bins[new_phi_0_b]=min(means_bcat_phi)
        """
        
        
        

    
    #print("find mean per bin phi --- %s seconds ---\n" % (time.time() - start_time_2))
    #means_ov_phi=feat_filt.groupby(['phi_bin_DV'])["betaCatenin_nuc"].mean()
    #phi_labs_abs[means_ov_phi.tolist().index(min(means_ov_phi))]
    new_phi_min=min(min_bins, key=min_bins.get)
    row_to_fill.append(new_phi_min)
    row_to_fill.append((new_phi_min-cur_phi_0)*180/math.pi)
    #results.loc[idx+1]=row_to_fill
    """
    #Model Max r2
    min_bins=dict(sorted(min_bins.items()))
    min_bins_values=list(min_bins.values())
    min_bins_keys=list(min_bins.keys())
    shifts=min_bins_keys+[0]
    #key_shift=list(min_bins.keys())[list(min_bins.values()).index(min(min_bins_values))]
    all_r2=[]
    for key_shift in range(len(shifts[:-1])):
        #shift_list=math.ceil(len(min_bins_values)/2)-1-min_bins_keys.index(key_shift)
        min_bins_values_shift=list(np.roll(min_bins_values, key_shift))
        model = np.poly1d(np.polyfit(shifts[:-1], min_bins_values_shift, 2))
        r2=r2_score(min_bins_values, model(shifts[:-1]))
        if r2<=0:
            r2=0
        all_r2.append(r2)
    max_r2_idx=all_r2.index(max(all_r2))
    min_bins_values=list(np.roll(min_bins_values, max_r2_idx))
    model = np.poly1d(np.polyfit(shifts[:-1], min_bins_values, 2)) 
    row_to_fill.append(r2_score(min_bins_values, model(shifts[:-1])))
    row_to_fill.append(model[2])    
    new_phi_min=np.poly1d.deriv(model).roots
    row_to_fill.append(new_phi_min[0])
    row_to_fill.append((new_phi_min[0]-cur_phi_0)*180/math.pi)
    polyline = np.linspace(min(shifts), max(shifts), 100)
    results.loc[idx+1]=row_to_fill
    fig, ax=plt.subplots()
    ax.scatter(shifts[:-1], min_bins_values)
    ax.plot(polyline, model(polyline))
    ax.set_title(str(cur_phi_0))
    fig.savefig(path_out_im+im+"_max_r2_model_"+str(n_bins)+".png", bbox_inches='tight', dpi=300) # 
    plt.close()
    
    """
    #Model Minimum
    min_bins=dict(sorted(min_bins.items()))
    min_bins_values=list(min_bins.values())
    min_bins_keys=list(min_bins.keys())
    key_shift=list(min_bins.keys())[list(min_bins.values()).index(min(min_bins_values))]
    shift_list=math.ceil(len(min_bins_values)/2)-1-min_bins_keys.index(key_shift)
    min_bins_values=list(np.roll(min_bins_values, shift_list))
    #min_bins_keys=list(np.roll(min_bins_keys, shift_list))
    #shifts_b=np.roll(shifts[:-1], shift_list)
    
    shifts=min_bins_keys+[0]
    model = np.poly1d(np.polyfit(shifts[:-1], min_bins_values, 2)) 
    row_to_fill.append(r2_score(min_bins_values, model(shifts[:-1])))
    row_to_fill.append(model[2])    
    new_phi_min=np.poly1d.deriv(model).roots
    row_to_fill.append(new_phi_min[0])
    row_to_fill.append((new_phi_min[0]-cur_phi_0)*180/math.pi)
    results.loc[idx+1]=row_to_fill
    polyline = np.linspace(min(shifts), max(shifts), 100)
    fig, ax=plt.subplots()
    ax.scatter(shifts[:-1], min_bins_values)
    ax.plot(polyline, model(polyline))
    ax.set_title(str(cur_phi_0))
    fig.savefig(path_out_im+im+"_min_model_"+str(n_bins)+".png", bbox_inches='tight', dpi=300) # 
    plt.close()
    
    """
    #Model Maximum
    min_bins_values=list(min_bins.values())
    min_bins_keys=list(min_bins.keys())
    key_shift=list(min_bins.keys())[list(min_bins.values()).index(max(min_bins_values))]
    shift_list=len(min_bins_values)-1-min_bins_keys.index(key_shift)
    min_bins_values=list(np.roll(min_bins_values, shift_list))
    
    shifts=min_bins_keys+[0]
    model = np.poly1d(np.polyfit(shifts[:-1], min_bins_values, 2)) 
    row_to_fill.append(r2_score(min_bins_values, model(shifts[:-1])))
    row_to_fill.append(model[2])    
    new_phi_min=np.poly1d.deriv(model).roots
    row_to_fill.append(new_phi_min)
    row_to_fill.append((new_phi_min-cur_phi_0)*180/math.pi)
    
    results.loc[idx+1]=row_to_fill
    polyline = np.linspace(min(shifts), max(shifts), 100)
    fig, ax=plt.subplots()
    ax.scatter(shifts[:-1], min_bins_values)
    ax.plot(polyline, model(polyline))
    ax.set_title(str(cur_phi_0))
    fig.savefig(path_out_im+im+"_max_model_"+str(n_bins)+".png", bbox_inches='tight', dpi=300) # 
    plt.close()
    """
    
results=results.sort_values("hpf")
print(results)
results.min_phi_diff.plot.hist(bins=(int(math.floor(results.min_phi_diff.max()-results.min_phi_diff.min())/12)))
plt.show()
    
    