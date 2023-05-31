#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:41:32 2023

@author: floriancurvaia
"""



import numpy as np

import time

import pandas as pd

import math

import seaborn as sns

import matplotlib.pyplot as plt

from pathlib import Path

from scipy.stats import zscore



start_time_0=time.time()

hpf="6"
n_bins=30
#plt.rcParams.update({'font.size': 22})
sns.set(style="ticks", font_scale=0.5)
#sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})  
#sns.set_theme(style='white')
#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/"
path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/"
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Heatmaps_avg/"

#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/pre_aligned/distances/"
#path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/pre_aligned/Images/Heatmaps_avg/"

stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio', 'pSmad2/3_nc_ratio']
#stains=['pSmad1/5_nuc', 'betaCatenin_nuc', 'MapK_nc_ratio', 'pSmad2/3_nc_ratio']
wells=["D06", "B07", "C07", "D07"] #6 hpf
#wells=["B08", "C08", "D08"] #7 hpf
#wells=["D05", "B06", "C06"] #5_7 hpf
#wells=["C05"] #5_3 hpf
#wells=["B05"] #4_7 hpf
#wells=["B04", "C04", "D04"] #4_3 hpf
#wells=["B03", "C03", "D03"] #3_7 hpf
#wells=["B02", "C02", "D02"] #3_3 hpf
fld=Path(path_in)
files=[]
all_df=[]
#to_remove=['D06_px+1741_py-0131', 'D06_px-1055_py-0118', 'B07_px+1257_py-0474', 'C07_px+1929_py-0176', "B07_px-0202_py+0631", "C07_px+0243_py-1998"]
to_remove=["B07_px-0202_py+0631", "C07_px+0243_py-1998"]
for w in wells:
    for f in fld.glob(w+"*"+"_w_dist_sph_simp.csv"):
        name=f.name
        emb=name.split("_w")[0]
        if emb in to_remove:
            pass
        else:
            try:
                files.append(emb)
                fn=path_in+emb+"_w_dist_sph_simp.csv"
                feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["rel_theta", "cur_phi"]+stains) #, "dist_out"
                feat_filt["emb"]=emb
                for stain in stains:
                    y_max=feat_filt[stain].quantile(0.975)
                    y_min=feat_filt[stain].quantile(0.025)
                    #feat_filt=feat_filt.loc[feat_filt[stain]>y_min]
                    #feat_filt=feat_filt.loc[feat_filt[stain]<y_max]
                    feat_filt[stain]=zscore(feat_filt[stain])
                #feat_filt["rel_theta"]=feat_filt.theta/np.max(feat_filt.theta)
                all_df.append(feat_filt)
                del feat_filt
            except ValueError:
                pass
            
        
        
#files.remove("C07_px+0243_py-1998")
#files.remove('B07_px-0202_py+0631')

print(len(all_df))
feat_filt=pd.concat(all_df)
#feat_filt=all_df[2]

labels = range(1, n_bins)
#theta_bins=np.linspace(0, math.pi, n_bins, endpoint=True)
theta_bins=np.linspace(0, 1, n_bins, endpoint=True)
phi_bins_abs=np.linspace(0, math.pi, n_bins)
phi_labs_abs=360*phi_bins_abs/(2*math.pi)
phi_labs_abs=phi_labs_abs[:-1]


phi_labels=[str(x) for x in np.around(phi_labs_abs, 2)]
phi_labels[0]="D "+6*" "+phi_labels[0]
#phi_labels[0]=phi_labels[0][::-1].expandtabs()[::-1]
phi_labels[-1]="V " +phi_labels[-1]
#phi_labels[-1]=phi_labels[-1][::-1].expandtabs()[::-1]

#phi_labels=["" for i in range(len(labels))]
#phi_labels[0]="D"
#phi_labels[-1]="V"

theta_labels=[str(x) for x in labels]
theta_labels[0]=theta_labels[0]+"\nAP"
theta_labels[-1]=theta_labels[-1]+"\nMargin"
#theta_labels[0]="AP\t"+phi_labels[0]
#phi_labels[-1]="Margin\t" +phi_labels[-1]


feat_filt['theta_bin'] = pd.to_numeric(pd.cut(x = feat_filt['rel_theta'], bins = theta_bins, labels = labels, include_lowest = True, right=True))

feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'].abs(), bins = phi_bins_abs, labels = labels, include_lowest = True, right=False))
#feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['phi'], bins = phi_bins_abs, labels = labels, include_lowest = True, right=False))

#feat_filt=feat_filt.loc[feat_filt.rel_theta>0.8]

print("load files --- %s seconds ---\n" % (time.time() - start_time_0))


start_time_1=time.time()
for stain in stains:
    feat_filt=feat_filt.drop(feat_filt[feat_filt[stain]==0].index)

fig, axs=plt.subplots(2,2, figsize=(10, 10))
fig.set_dpi(300)
fig.subplots_adjust(hspace=0.2)
fig.subplots_adjust(wspace=0.15)


fig1, axs1=plt.subplots(2,2, figsize=(10, 10))
fig1.set_dpi(300)
fig1.subplots_adjust(hspace=0.15)
fig1.subplots_adjust(wspace=0.15)

fig2, axs2=plt.subplots(2,2, figsize=(10, 10))
fig2.set_dpi(300)
fig2.subplots_adjust(hspace=0.15)
fig2.subplots_adjust(wspace=0.15)

for stain, ax_, ax_1, ax_2 in zip(stains, axs.flatten(), axs1.flatten(), axs2.flatten()):
    c=stain.split("_")
    stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
    #Phi Theta heatmaps
    
    
    means_phi_theta=feat_filt.groupby(['phi_bin_abs', 'theta_bin'])[stain].mean()
    
    heatmap_phi_theta=np.full([n_bins-1, n_bins-1], np.nan)
    
    for idx in means_phi_theta.index:
        idx_phi=idx[0]
        idx_theta=idx[1]
        heatmap_phi_theta[idx_phi-1, idx_theta-1]=means_phi_theta.loc[(idx_phi,idx_theta)]
        
    #fig=plt.figure()
    
    vmin=np.nanquantile(heatmap_phi_theta, 0.01)
    vmax=np.nanquantile(heatmap_phi_theta, 0.99)
    #vmin=feat_filt[stain].min()
    #vmax=feat_filt[stain].max()
    #vmin=np.nanmin(heatmap_phi_theta)
    #vmax=np.nanmax(heatmap_phi_theta)
    
    ax=sns.heatmap(heatmap_phi_theta, linewidth=0.1, vmin=vmin, vmax=vmax, xticklabels=theta_labels, yticklabels=phi_labels, cmap=sns.mpl_palette("viridis", 1000), cbar=True, ax=ax_) #xticklabels=labels, yticklabels=np.around(phi_labs_abs, 2),
    #ax.set_xticks(labels, labels)
    #ax.set_yticks(labels, np.around(phi_labs_abs, 2))
    ax.collections[0].colorbar.set_label(stain+" intensity")
    ax_.set_xlabel("theta bin", va='bottom')
    ax_.set_ylabel("phi", va='top')
    ax_.set_xticks(np.array(labels)-0.5, theta_labels, rotation=0)
    
   
    
    
    
    means_ov_phi=feat_filt.groupby(['phi_bin_abs'])[stain].mean()

    ax_1.plot(phi_labs_abs, means_ov_phi)
    ax_1.set_xlabel("phi")
    ax_1.set_ylabel(stain)
    

    means_ov_theta=feat_filt.groupby(['theta_bin'])[stain].mean()

    #fig5=plt.figure()
    #plt.clf()
    ax_2.plot(theta_bins[:-1], means_ov_theta)
    ax_2.set_xlabel("theta")
    ax_2.set_ylabel(stain)
    


fig.savefig(path_out_im+str(hpf)+"_hpf_phi_abs_theta.png", bbox_inches='tight', dpi=300)
plt.close(fig)

fig1.savefig(path_out_im+str(hpf)+"_hpf_means_ov_phi.png", bbox_inches='tight', dpi=300)
plt.close(fig1)

fig2.savefig(path_out_im+str(hpf)+"_hpf_means_ov_theta.png", bbox_inches='tight', dpi=300)
plt.close(fig2)
print("Generate heatmaps --- %s seconds ---\n" % (time.time() - start_time_1))



