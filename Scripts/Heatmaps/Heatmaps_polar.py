#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:17:13 2023

@author: floriancurvaia
"""

import numpy as np
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
import pandas as pd
from pathlib import Path
import math

#plt.ioff()
path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Ortho_proj/"
path_in_df="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/"

n_bins=20
ref_emb={7.0:"B08_px+1076_py-0189", 6.0:"C07_px-0784_py-0616", 5.7:'D05_px+1922_py+0876', 5.3:'C05_px+0198_py+1683', 4.7:'B05_px+1522_py-1087',
         4.3:'C04_px-0816_py-1668', 3.7:'C03_px-0480_py-1856', 3.3:'B02_px+1709_py+0973'}
time_emb={"B02":3.3, "C02":3.3, "D02":3.3, "B03":3.7, "C03":3.7, "D03":3.7, "B04":4.3, "C04":4.3, "D04":4.3,
          "B05":4.7, "C05":5.3, "D05":5.7, "B06":5.7, "C06":5.7, "D06":6.0, "B07":6.0, "C07":6.0, "D07":6.0,
          "B08":7.0, "C08":7.0, "D08":7.0}

#wells=["D06", "B07", "C07", "D07"] #6 hpf
#wells=["B08", "C08", "D08"] #7 hpf
#wells=["D05", "B06", "C06"] #5_7 hpf
#wells=["C05"] #5_3 hpf
#wells=["B05"] #4_7 hpf
#wells=["B04", "C04", "D04"] #4_3 hpf
#wells=["B03", "C03", "D03"] #3_7 hpf
wells=["B02", "C02", "D02"] #3_3 hpf

hpf=time_emb[wells[0]]

dir_hpf=Path(path_out_im+str(hpf).replace(".", "_")+"_hpf")
dir_hpf.mkdir(exist_ok=True)
path_out_im=dir_hpf.as_posix()+"/"

ax_labels=range(1, n_bins+1)
theta_bins=np.linspace(0, 1, n_bins+1, endpoint=True)
phi_bins=np.linspace(-math.pi, math.pi, 2*n_bins+1)
phi_labs=range(1,2*n_bins+1)
phi_bins_abs=np.linspace(0, math.pi, n_bins+1)
phi_labs_abs=360*phi_bins_abs/(2*math.pi)
phi_labs_abs=phi_labs_abs[:-1]

#im=ref_emb[hpf]+"_w_dist_sph_simp.csv"

#feat_filt=pd.read_csv(path_in_df + im, sep=",", index_col=False)

fld=Path(path_in_df)
files=[]
all_df=[]
stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio']
#stain="betaCatenin_nuc"
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
                fn=path_in_df+emb+"_w_dist_sph_simp.csv"
                feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["rel_theta", "cur_phi", "dist_out"]+stains)
                files.append(emb)
                feat_filt["emb"]=emb
                #y_max=feat_filt[stain].quantile(0.99)
                #y_min=feat_filt[stain].quantile(0.025)
                #feat_filt=feat_filt.loc[feat_filt[stain]>y_min]
                #feat_filt=feat_filt.loc[feat_filt[stain]<y_max]
                #feat_filt["rel_theta"]=feat_filt.theta/np.max(feat_filt.theta)
                #feat_filt['theta_bin_pos'] = pd.qcut(feat_filt['rel_theta'], n_bins_per_ax, labels=False)+1
                feat_filt['theta_bin'] = pd.to_numeric(pd.cut(x = feat_filt['rel_theta'], bins = theta_bins, labels = ax_labels, include_lowest = False, right=True))
                feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'].abs(), bins = phi_bins_abs, labels = ax_labels, include_lowest = False, right=True))
                feat_filt['phi_bin'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'], bins = phi_bins, labels = phi_labs, include_lowest = False, right=True))
                feat_filt=feat_filt.sort_values(["phi_bin_abs", "theta_bin"])
                feat_filt.reset_index(inplace=True)
                all_df.append(feat_filt)
                #del feat_filt
            except ValueError:
                pass


#feat_filt['theta_bin'] = pd.to_numeric(pd.cut(x = feat_filt['rel_theta'], bins = theta_bins, labels = ax_labels, include_lowest = False, right=True))
#feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'].abs(), bins = phi_bins_abs, labels = ax_labels, include_lowest = False, right=True))
#feat_filt['phi_bin'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'], bins = phi_bins, labels = phi_labs, include_lowest = False, right=True))

feat_filt_all=pd.concat(all_df)
feat_filt=feat_filt_all.loc[feat_filt_all.emb==ref_emb[hpf]]

coords=["rel_theta", "cur_phi", "dist_out"]
#ticks_lab=np.array(np.linspace(0, 360, 8, endpoint=False).astype(int), dtype="str")
#ticks_lab[0]=ticks_lab[0]+"\nDorsal"
#ticks_lab[4]=ticks_lab[4]+"\nVentral"

feat_filt=feat_filt.sort_values(coords[2])
for coord in coords:
    fig1, ax1 = plt.subplots(figsize=(7.5,7.5), subplot_kw={'projection': 'polar'})
    #feat_filt=feat_filt.sort_values(coord)
    scatter_coord=ax1.scatter(feat_filt.cur_phi,feat_filt.rel_theta,c=feat_filt[coord], edgecolors='face', cmap="turbo", marker="o", s=15)
    ax1.grid(linewidth=0.5)
    ax1.set_yticklabels(ax1.get_yticklabels(), weight='bold')
    #ax1.set_xticklabels(ticks_lab)
    ax1.set_ylim([0,1])
    ax1.set_xticks(np.pi/180. * np.linspace(180,  -180, 8, endpoint=False))
    ax1.set_thetalim(-np.pi, np.pi)
    #fig1.colorbar(scatter_coord, ax=ax1, label=coord, shrink=.85)
    fig1.savefig(path_out_im+str(hpf)+"_polar_scatter_"+coord+".png", dpi=300) #bbox_inches='tight',
    plt.close()


"""
fig, ax=plt.subplots(figsize=(10,10))

ax.scatter(feat_filt.x_corr, feat_filt.y_corr, c=feat_filt[stain],vmin=feat_filt[stain].quantile(0.05), vmax=feat_filt[stain].quantile(0.95))
ax.set_aspect('equal', adjustable='box')
"""

"""   
fig, ax=plt.subplots(figsize=(10,10))

ax.scatter(feat_filt.x_corr, feat_filt.y_corr, c=feat_filt[stain+"_mean"]) #,vmin=feat_filt[stain].quantile(0.05), vmax=feat_filt[stain].quantile(0.95)
ax.set_aspect('equal', adjustable='box')
"""

for stain in stains:
    c=stain.split("_")
    stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
    mean_per_emb=np.zeros((len(files), 2*n_bins, n_bins))
    
    CV_per_emb=np.zeros((len(files), 2*n_bins, n_bins))
    
    for emb, i in zip(files, range(len(files))):
        feat_filt_emb=feat_filt_all.loc[feat_filt_all.emb==emb]
        
        means_emb_phi_theta=feat_filt_emb.groupby(['phi_bin', 'theta_bin'])[stain].mean()
        
        heatmap_emb_phi_theta=np.full([2*n_bins, n_bins], np.nan)
        for idx in means_emb_phi_theta.index:
            idx_phi=idx[0]
            idx_theta=idx[1]
            heatmap_emb_phi_theta[idx_phi-1, idx_theta-1]=means_emb_phi_theta.loc[(idx_phi,idx_theta)]
        mean_per_emb[i]=heatmap_emb_phi_theta
        
        CV_emb_phi_theta=feat_filt_emb.groupby(['phi_bin', 'theta_bin'])[stain].std()/feat_filt_emb.groupby(['phi_bin', 'theta_bin'])[stain].mean()
        
        CV_emb_heatmap_phi_theta=np.full([2*n_bins, n_bins], np.nan)
        for idx in CV_emb_phi_theta.index:
            idx_phi=idx[0]
            idx_theta=idx[1]
            CV_emb_heatmap_phi_theta[idx_phi-1, idx_theta-1]=CV_emb_phi_theta.loc[(idx_phi,idx_theta)]
            
        CV_per_emb[i]=CV_emb_heatmap_phi_theta
        
        #vmin=np.nanquantile(heatmap_phi_theta, 0.01)
        #vmax=np.nanquantile(heatmap_phi_theta, 0.99)
        """
        fig1, ax1 = plt.subplots(figsize=(7.5,7.5), subplot_kw={'projection': 'polar'})
        #ax2.scatter(feat_filt.cur_phi,feat_filt.rel_theta,c=feat_filt[stain],edgecolors='face', vmin=feat_filt[stain].quantile(0.05), vmax=feat_filt[stain].quantile(0.95))
        #ax3.pcolormesh(feat_filt.cur_phi,feat_filt.rel_theta,feat_filt[stain],edgecolors='face')
        hm_CV_emb=ax1.pcolormesh(phi_bins, theta_bins, CV_emb_heatmap_phi_theta.T, cmap="turbo", edgecolors='face', vmin=0, vmax=0.5) #, vmin=vmin, vmax=vmax
        #ax3.pcolormesh(-1*phi_bins_abs, theta_bins, heatmap_phi_theta.T, cmap="turbo", edgecolors='face')
        ax1.grid(linewidth=0.5)
        ax1.set_xticks(np.pi/180. * np.linspace(180,  -180, 8, endpoint=False))
        ax1.set_thetalim(-np.pi, np.pi)
        fig1.colorbar(hm_CV_emb, ax=ax1, label=stain_clean, shrink=.85)
        fig1.savefig(path_out_im+"polar_CV_"+emb+"_"+stain_clean+".png", dpi=300)
        plt.close()
        """
    #feat_filt=feat_filt.sort_values(stain)
    
    fig2, ax2 = plt.subplots(figsize=(7.5,7.5), subplot_kw={'projection': 'polar'})
    scatter_stain=ax2.scatter(feat_filt.cur_phi,feat_filt.rel_theta,c=feat_filt[stain], edgecolors='face', 
                vmin=feat_filt[stain].quantile(0.05), vmax=feat_filt[stain].quantile(0.95), cmap="turbo",
                marker="o", s=15)
    ax2.grid(linewidth=0.5)
    ax2.set_yticklabels(ax2.get_yticklabels(), weight='bold')
    ax2.set_ylim([0,1])
    ax2.set_xticks(np.pi/180. * np.linspace(180,  -180, 8, endpoint=False))
    ax2.set_thetalim(-np.pi, np.pi)
    #cax = fig2.add_axes([ax2.get_position().x1+0.01,ax2.get_position().y0,0.02,ax2.get_position().height])
    #cax = fig2.add_axes([ax2.get_position().x1-0.25,ax2.get_position().y0,0.02,ax2.get_position().y1-ax2.get_position().y0])
    fig2.colorbar(scatter_stain, ax=ax2, label=stain_clean, shrink=.85)
    fig2.savefig(path_out_im+"polar_scatter_"+stain_clean+".png", dpi=300) #bbox_inches='tight', pad_inches=0
    plt.close()
    #ax2.pcolormesh(feat_filt.cur_phi,feat_filt.rel_theta,feat_filt[stain],edgecolors='face')
    #ax1.imshow(heatmap_phi_theta)
    
    """
    means_phi_abs_theta=feat_filt_all.groupby(['phi_bin_abs', 'theta_bin'])[stain].mean()
    
    heatmap_phi_abs_theta=np.full([n_bins, n_bins], np.nan)
    for idx in means_phi_abs_theta.index:
        idx_phi=idx[0]
        idx_theta=idx[1]
        heatmap_phi_abs_theta[idx_phi-1, idx_theta-1]=means_phi_abs_theta.loc[(idx_phi,idx_theta)]
        #feat_filt.loc[(feat_filt.phi_bin_abs==idx_phi) & (feat_filt.theta_bin==idx_theta), stain+"_mean"]=means_phi_abs_theta.loc[(idx_phi,idx_theta)]
    
    fig3, ax3 = plt.subplots(figsize=(10,10), subplot_kw={'projection': 'polar'})
    #ax2.scatter(feat_filt.cur_phi,feat_filt.rel_theta,c=feat_filt[stain],edgecolors='face', vmin=feat_filt[stain].quantile(0.05), vmax=feat_filt[stain].quantile(0.95))
    #ax3.pcolormesh(feat_filt.cur_phi,feat_filt.rel_theta,feat_filt[stain],edgecolors='face')
    ax3.pcolormesh(phi_bins_abs, theta_bins, heatmap_phi_abs_theta.T, cmap="turbo", edgecolors='face')
    ax3.pcolormesh(-1*phi_bins_abs, theta_bins, heatmap_phi_abs_theta.T, cmap="turbo", edgecolors='face')
    ax3.grid(linewidth=0.5)
    fig3.savefig(path_out_im+str(hpf)+"_polar_mean_phi_abs_"+stain_clean+".png", dpi=300)
    plt.close()
    """
    
    means_phi_theta=feat_filt_all.groupby(['phi_bin', 'theta_bin'])[stain].mean()
    
    heatmap_phi_theta=np.full([2*n_bins, n_bins], np.nan)
    for idx in means_phi_theta.index:
        idx_phi=idx[0]
        idx_theta=idx[1]
        heatmap_phi_theta[idx_phi-1, idx_theta-1]=means_phi_theta.loc[(idx_phi,idx_theta)]
    
    vmin=np.nanquantile(heatmap_phi_theta, 0.01)
    vmax=np.nanquantile(heatmap_phi_theta, 0.99)
    
    fig4, ax4 = plt.subplots(figsize=(7.5,7.5), subplot_kw={'projection': 'polar'})
    #ax2.scatter(feat_filt.cur_phi,feat_filt.rel_theta,c=feat_filt[stain],edgecolors='face', vmin=feat_filt[stain].quantile(0.05), vmax=feat_filt[stain].quantile(0.95))
    #ax3.pcolormesh(feat_filt.cur_phi,feat_filt.rel_theta,feat_filt[stain],edgecolors='face')
    hm_mean=ax4.pcolormesh(phi_bins, theta_bins, heatmap_phi_theta.T, cmap="turbo", edgecolors='face', vmin=vmin, vmax=vmax)
    #ax3.pcolormesh(-1*phi_bins_abs, theta_bins, heatmap_phi_theta.T, cmap="turbo", edgecolors='face')
    ax4.grid(linewidth=0.5)
    ax4.set_xticks(np.pi/180. * np.linspace(180,  -180, 8, endpoint=False))
    ax4.set_thetalim(-np.pi, np.pi)
    fig4.colorbar(hm_mean, ax=ax4, label=stain_clean, shrink=.85)
    fig4.savefig(path_out_im+"polar_mean_"+stain_clean+".png", dpi=300) #, bbox_inches="tight"
    plt.close()
    
    """
    CV_phi_theta=feat_filt_all.groupby(['phi_bin', 'theta_bin'])[stain].std()/feat_filt_all.groupby(['phi_bin', 'theta_bin'])[stain].mean()
    
    CV_heatmap_phi_theta=np.full([2*n_bins, n_bins], np.nan)
    for idx in CV_phi_theta.index:
        idx_phi=idx[0]
        idx_theta=idx[1]
        CV_heatmap_phi_theta[idx_phi-1, idx_theta-1]=CV_phi_theta.loc[(idx_phi,idx_theta)]
    """
    #vmin=np.nanquantile(heatmap_phi_theta, 0.01)
    #vmax=np.nanquantile(heatmap_phi_theta, 0.99)
    
    CV_heatmap_phi_theta=np.nanstd(mean_per_emb, axis=0)/np.nanmean(mean_per_emb, axis=0)
    
    fig5, ax5 = plt.subplots(figsize=(7.5,7.5), subplot_kw={'projection': 'polar'})
    #ax2.scatter(feat_filt.cur_phi,feat_filt.rel_theta,c=feat_filt[stain],edgecolors='face', vmin=feat_filt[stain].quantile(0.05), vmax=feat_filt[stain].quantile(0.95))
    #ax3.pcolormesh(feat_filt.cur_phi,feat_filt.rel_theta,feat_filt[stain],edgecolors='face')
    hm_CV=ax5.pcolormesh(phi_bins, theta_bins, CV_heatmap_phi_theta.T, cmap="turbo", edgecolors='face', vmin=0, vmax=0.5) #, vmin=vmin, vmax=vmax
    #ax3.pcolormesh(-1*phi_bins_abs, theta_bins, heatmap_phi_theta.T, cmap="turbo", edgecolors='face')
    ax5.grid(linewidth=0.5)
    ax5.set_xticks(np.pi/180. * np.linspace(180,  -180, 8, endpoint=False))
    ax5.set_thetalim(-np.pi, np.pi)
    fig5.colorbar(hm_CV, ax=ax5, label=stain_clean, shrink=.85)
    fig5.savefig(path_out_im+"polar_CV_inter_embs_"+stain_clean+".png", dpi=300) #
    plt.close()
    
    fig6, ax6=plt.subplots()
    ax6.hist(CV_per_emb.flatten(), bins=100)
    fig6.savefig(path_out_im+"hist_CV_intra_embs_"+stain_clean+".png", dpi=300)
    plt.close()
    
    
    fig7, ax7=plt.subplots()
    ax7.hist(CV_heatmap_phi_theta.flatten(), bins=100)
    fig7.savefig(path_out_im+"hist_CV_inter_embs_"+stain_clean+".png", dpi=300)
    plt.close()
    
    fig8, ax8 = plt.subplots(figsize=(7.5,7.5), subplot_kw={'projection': 'polar'})
    #ax2.scatter(feat_filt.cur_phi,feat_filt.rel_theta,c=feat_filt[stain],edgecolors='face', vmin=feat_filt[stain].quantile(0.05), vmax=feat_filt[stain].quantile(0.95))
    #ax3.pcolormesh(feat_filt.cur_phi,feat_filt.rel_theta,feat_filt[stain],edgecolors='face')
    hm_CV_intra=ax8.pcolormesh(phi_bins, theta_bins, np.nanmean(CV_per_emb, axis=0).T, cmap="turbo", edgecolors='face', vmin=0, vmax=0.5) #, vmin=vmin, vmax=vmax
    #ax3.pcolormesh(-1*phi_bins_abs, theta_bins, heatmap_phi_theta.T, cmap="turbo", edgecolors='face')
    ax8.grid(linewidth=0.5)
    ax8.set_xticks(np.pi/180. * np.linspace(180,  -180, 8, endpoint=False))
    ax8.set_thetalim(-np.pi, np.pi)
    fig8.colorbar(hm_CV_intra, ax=ax8, label=stain_clean, shrink=.85)
    fig8.savefig(path_out_im+"polar_CV_intra_embs_"+stain_clean+".png", dpi=300) #
    plt.close()
    












