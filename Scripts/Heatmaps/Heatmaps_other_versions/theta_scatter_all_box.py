#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:44:36 2023

@author: floriancurvaia
"""


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from math import pi

import numba


#B02_px-0280_py+1419, B03_px-0545_py-1946, B04_px+0114_py-1436, B05_px+1522_py-1087, C05_px+0198_py+1683, 
#D05_px-0738_py-1248, B06_px-0493_py+0146, D06_px-1055_py-0118, B07_px+1257_py-0474, B08_px+1076_py-0189
#C04_px-0816_py-1668 INSTEAD OF B04




images=["B02_px-0280_py+1419", "B03_px-0545_py-1946", "C04_px-0816_py-1668", "B05_px+1522_py-1087", "C05_px+0198_py+1683", 
        "D05_px-0738_py-1248", "B06_px-0493_py+0146", "D06_px-1055_py-0118", "B07_px+1257_py-0474", "B08_px+1076_py-0189"]

#['betaCatenin_0_Mean', 'pMyosin_1_Mean', 'MapK_2_Mean', 'pSmad2/3_2_Mean', 'pSmad1/5_4_Mean', 'Pol2-S5P_4_Mean', 'mTOR-pS6-H3K27AC_5_Mean']
#stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio', 'pSmad2/3_nc_ratio'] #'betaCatenin_nc_ratio'

stains=['betaCatenin_nuc', 'pMyosin_nc_ratio', 'MapK_nc_ratio', 'pSmad2/3_nc_ratio', 'pSmad1/5_nc_ratio', 'Pol2-S5P_nc_ratio', 'mTOR-pS6-H3K27AC_nc_ratio']

#stains=['pSmad1/5_nuc', 'betaCatenin_nuc', 'MapK_nuc', 'pSmad2/3_nuc'] #'betaCatenin_nc_ratio'

time_emb={"B02":3.3, "C02":3.3, "D02":3.3, "B03":3.7, "C03":3.7, "D03":3.7, "B04":4.3, "C04":4.3, "D04":4.3,
          "B05":4.7, "C05":5.3, "D05":5.7, "B06":5.7, "C06":5.7, "D06":6.0, "B07":6.0, "C07":6.0, "D07":6.0,
          "B08":7.0, "C08":7.0, "D08":7.0}

path_in_adj="/data/homes/fcurvaia/Spheres_fit/"

path_out_im="/data/homes/fcurvaia/Images/Theta_scatter/"

@numba.jit(nopython=True, parallel=True, fastmath=True)
def neigh_med(adj_mat, stain):
    neighbours_median=np.zeros((stain.shape[0]))
    neigh_val=adj_mat*stain
    for i in numba.prange(adj_mat.shape[1]):
        a=neigh_val[i]
        neighbours_median[i]=np.median(a[np.nonzero(a)])
        
    return neighbours_median

def MAD(data):
    median = np.median(data)
    diff   = abs(data-median)
    MAD = np.median(diff)
    return MAD

n_bins=30
n_bins_cv=20
all_df=[]
adj_mat_all={}
for im in images:
    fn1="/data/homes/fcurvaia/distances/"+im+"_w_dist_sph_simp.csv"
    #feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["Label", "Centroid_x", "Centroid_y", "Centroid_z", "x_corr", "y_corr", "z_corr", "structure", "r_neigh", "r_neigh_mean", "NTouchingNeighbors","PhysicalSize", "theta","phi", "phi_bin", "theta_bin", "dist_out", "phi_bin_new", "phi_bin_cur", "new_phi", "cur_phi"]+stains)
    feat_filt=pd.read_csv(fn1, sep=",", index_col=False, usecols=["Label", "r_neigh", "NTouchingNeighbors","PhysicalSize", "theta", "dist_out", "cur_phi", "phi_bin_cur", "theta_bin"]+stains)
    well=im.split("_")[0]
    feat_filt["hpf"]=time_emb[well]
    feat_filt["emb"]=well
    """
    for stain in stains:
        percentile_top=np.percentile(feat_filt[stain],97.5)
        percentile_bot=np.percentile(feat_filt[stain],2.5)
        feat_filt=feat_filt[feat_filt[stain] < percentile_top]
        feat_filt=feat_filt[feat_filt[stain] > percentile_bot]
    """
    all_df.append(feat_filt)
    adj_mat_all[well]=np.load(path_in_adj+im+"_adj_mat.npy")
    
feat_filt_all=pd.concat(all_df)

for stain in stains:
    c=stain.split("_")
    stain_clean="_".join(["".join(c[0].split("/"))]+c[1:])
    
    plt.style.use('dark_background')
    f1, axs1 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.25]}, figsize=(60, 30))
    fon_size=15
    #space_size=38
    plt.rcParams.update({'font.size': fon_size}) 
    f1.subplots_adjust(hspace=0.125)
    f1.subplots_adjust(wspace=0.15)
    f1.set_dpi(300)
    f2, axs2 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.25]}, figsize=(60, 30))
    f2.subplots_adjust(hspace=0.125)
    f2.subplots_adjust(wspace=0.125)
    f2.set_dpi(300)
    f3, axs3 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.25]}, figsize=(60, 30))
    f3.subplots_adjust(hspace=0.125)
    f3.subplots_adjust(wspace=0.125)
    f3.set_dpi(300)
    
    l=len(images)

    
    if stain==stains[0]:
        f4, axs4 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(60, 30))
        f4.subplots_adjust(hspace=0.15)
        f4.subplots_adjust(wspace=0.15)
        f4.set_dpi(300)
        
    

    f5, axs5 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(60, 30), sharey=True)
    f5.subplots_adjust(hspace=0.125)
    f5.subplots_adjust(wspace=0.125)
    f5.set_dpi(300)
    
    
    f6, axs6 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(60, 30), sharey=True)
    f6.subplots_adjust(hspace=0.15)
    f6.subplots_adjust(wspace=0.125)
    f6.set_dpi(300)
        
    for im, ax, ax1, ax2, ax3, ax4, ax5 in zip(images, axs1.flatten(), axs2.flatten(), axs3.flatten(), axs4.flatten(), axs5.flatten(), axs6.flatten()):
        print(stain+" "+im)
        well=im.split("_")[0]
        hpf=time_emb[well]
        feat_filt=feat_filt_all.loc[feat_filt_all.emb==well]
        #adj_mat=np.load(path_in_adj+im+"_adj_mat.npy")
        adj_mat=adj_mat_all[well]
        #stain_neigh=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 1, np.multiply(adj_mat, feat_filt[stain].to_numpy()))
        stain_neigh=neigh_med(adj_mat, feat_filt[stain].to_numpy())
        phi_bins=np.linspace(-pi, pi, n_bins_cv)
        labels = range(1, n_bins_cv)
        theta_bins=np.linspace(0, pi, n_bins_cv, endpoint=True)

        feat_filt['phi_bin_cv'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'], bins = phi_bins, labels = labels, include_lowest = True, right=False))
        feat_filt['theta_bin_cv'] = pd.to_numeric(pd.cut(x = feat_filt['theta'], bins = theta_bins, labels = labels, include_lowest = True, right=False))
        
        phi_bins_abs=np.linspace(0, pi, n_bins_cv)
        feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'].abs(), bins = phi_bins_abs, labels = labels, include_lowest = True, right=False))
        
        t_max=np.max(feat_filt.theta_bin_cv)
        theta_labs=360*theta_bins/(2*pi)
        theta_labs=theta_labs[:t_max]
        phi_labs_abs=360*phi_bins_abs/(2*pi)
        phi_labs_abs=phi_labs_abs[:-1]
        
        phi_labs=360*phi_bins/(2*pi)
        phi_labs=phi_labs[:-1]



        #cv_pop=np.zeros((n_bins_cv-1,np.max(feat_filt.theta_bin_cv)))
        bins=feat_filt.groupby(['phi_bin_cv', 'theta_bin_cv'])[stain]
        phi_theta_bins_cv=bins.std()/bins.mean()
        cv_pop=phi_theta_bins_cv.unstack().values[:, :]
        #phi_theta_bins_cv=feat_filt.groupby(['phi_bin_cv', 'theta_bin_cv'])[stain].std()/feat_filt.groupby(['phi_bin_cv', 'theta_bin_cv'])[stain].mean()
        #for i in range(n_bins_cv):
        #    for j in range(n_bins_cv):
        #        if (i,j) in phi_theta_bins_cv.index:
        #            cv_pop[i-1, j-1]=phi_theta_bins_cv.loc[i, j]

        ax_=ax.imshow(cv_pop, cmap="turbo", vmin=0, vmax=1, aspect="auto")
        if well=="C05" or well=="B08":
            f1.colorbar(ax_, ax=ax, label="CV")
        ax.set_ylabel("phi bin")
        ax.set_xlabel("theta bin")
        ax.set_title(im+" "+str(hpf)+" hpf")
        ax.set_xticks(np.unique(feat_filt.theta_bin_cv), labels=np.around(theta_labs, 2), rotation=90)
        ax.set_yticks(np.unique(feat_filt.phi_bin_cv), labels=np.around(phi_labs, 2))
        
        """
        if well=="C05":
            feat_filt[feat_filt['theta_bin_cv']!=10].boxplot(column=stain, by='theta_bin_cv', ax=ax4)
            ax4.set_xticks(np.unique(feat_filt.theta_bin_cv)[:-1], labels=np.around(theta_labs, 2)[:-1], rotation=90)
            ax4.set_xlabel("Theta_bin")
            ax4.set_ylabel(stain+" median")
            ax4.set_title(im+" "+str(hpf)+" hpf")
            f5.suptitle('')
            
            
           
        
        else:
            feat_filt.boxplot(column=stain, by="theta_bin_cv", ax=ax4)
            ax4.set_xticks(np.unique(feat_filt.theta_bin_cv), labels=np.around(theta_labs, 2), rotation=90)
            ax4.set_xlabel("Theta_bin")
            ax4.set_ylabel(stain+" median")
            ax4.set_title(im+" "+str(hpf)+" hpf")
            f5.suptitle('')
            
        """
        
        feat_filt.boxplot(column=stain, by="theta_bin_cv", ax=ax4, showfliers=False, grid=False, color=dict(boxes='w', whiskers='w', medians='r', caps='w'))
        ax4.set_xticks(np.unique(feat_filt.theta_bin_cv), labels=np.around(theta_labs, 2), rotation=90)
        ax4.set_xlabel("Theta_bin")
        ax4.set_ylabel(stain+" median")
        ax4.set_title(im+" "+str(hpf)+" hpf")
        f5.suptitle('')
        
        
        feat_filt.boxplot(column=stain, by="phi_bin_cv", ax=ax5, showfliers=False, grid=False, color=dict(boxes='w', whiskers='w', medians='r', caps='w'))
        ax5.set_xticks(np.unique(feat_filt.phi_bin_cv), labels=np.around(phi_labs, 2), rotation=90)
        ax5.set_xlabel("Phi_bin")
        ax5.set_ylabel(stain+" median")
        ax5.set_title(im+" "+str(hpf)+" hpf")
        f6.suptitle('')
        
        if stain==stains[0]:
            bins_size=bins.size()
            bins_pop_cv=bins_size.unstack().values[:, :]
            
            ax_3=ax3.imshow(bins_pop_cv, cmap="turbo", aspect="auto")
            ax3.set_ylabel("phi bin")
            ax3.set_xlabel("theta bin")
            ax3.set_title(im+" "+str(hpf)+" hpf")
            ax3.set_xticks(np.unique(feat_filt.theta_bin_cv), labels=np.around(theta_labs, 2), rotation=90)
            ax3.set_yticks(np.unique(feat_filt.phi_bin_cv), labels=np.around(phi_labs, 2))
            f4.colorbar(ax_3, ax=ax3, label="N")

        
        
        if stain.split("_")[1]=="nuc":
            feat_filt[stain+"_neigh"]=(feat_filt[stain]-stain_neigh)/feat_filt[stain]
            feat_filt.loc[feat_filt[stain]<1, stain]=1
        else:
            feat_filt[stain+"_neigh"]=feat_filt[stain]-stain_neigh
        
        #feat_filt[stain+"_neigh"]=(feat_filt[stain]-stain_neigh)/feat_filt[stain]
        
        feat_filt.cur_phi=feat_filt.cur_phi.abs()
        y_min=feat_filt_all[stain].quantile(0.01)
        y_max=feat_filt_all[stain].quantile(0.99)
        
        ax_1=ax1.scatter(
            x=feat_filt.theta, y=feat_filt[stain],
            c=feat_filt["cur_phi"], vmin=0, vmax=pi, marker="o", s=16, cmap="jet", alpha=0.5)
        ax1.set_title(im+" "+str(hpf)+" hpf")
        if well=="C05" or well=="B08":
            f2.colorbar(ax_1, ax=ax1, label="cur_phi")
        
        ax1.set_xlabel("theta")
        ax1.set_ylabel(stain)
        ax1.set_ylim(y_min,y_max)
        ax1.set_xticks(theta_bins[:t_max], labels=np.around(theta_labs, 2))
        
        ax_2=ax2.scatter(
            x=feat_filt.theta, y=feat_filt[stain+"_neigh"],
            c=feat_filt["cur_phi"], vmin=0, vmax=pi, marker="o", s=16, cmap="jet", alpha=0.5)
        ax2.set_title(im+" "+str(hpf)+" hpf")
        if well=="C05" or well=="B08":
            f3.colorbar(ax_2, ax=ax2, label="cur_phi")
        
        ax2.set_xlabel("theta")
        ax2.set_ylabel(stain+"_shift")
        ax2.set_ylim(-1,1.5)
        ax2.set_xticks(theta_bins[:t_max], labels=np.around(theta_labs, 2))
        
    

    f5.savefig(path_out_im+stain_clean+"_median_ov_theta_box.png", bbox_inches='tight', dpi=300)

    f6.savefig(path_out_im+stain_clean+"_median_ov_phi_box.png", bbox_inches='tight', dpi=300)

    f1.savefig(path_out_im+stain_clean+"_bins_cv.png", bbox_inches='tight', dpi=300)
    f2.savefig(path_out_im+stain_clean+"_scatter.png", bbox_inches='tight', dpi=300)
    f3.savefig(path_out_im+stain_clean+"_scatter_median.png", bbox_inches='tight', dpi=300)
    
    if stain==stains[0]:
        f4.savefig(path_out_im+"bins_cv_size.png", bbox_inches='tight', dpi=300)
    
    plt.close()









