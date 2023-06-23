#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:35:17 2023

@author: floriancurvaia
"""




import numpy as np

import pandas as pd

from scipy.stats import zscore

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import math

import time

from pathlib import Path

from itertools import product

import tensorflow as tf
# tf.enable_eager_execution()
import tensorflow_probability as tfp

dist = tfp.distributions


start_time_0=time.time()


images=["B02_px-0280_py+1419", "B03_px-0545_py-1946", "C04_px-0816_py-1668", "B05_px+1522_py-1087", "C05_px+0198_py+1683", 
        "D05_px-0738_py-1248", "B06_px-0493_py+0146", "D06_px-1055_py-0118", "B07_px+1257_py-0474", "B08_px+1076_py-0189"]


time_emb={"B02":3.3, "C02":3.3, "D02":3.3, "B03":3.7, "C03":3.7, "D03":3.7, "B04":4.3, "C04":4.3, "D04":4.3,
          "B05":4.7, "C05":5.3, "D05":5.7, "B06":5.7, "C06":5.7, "D06":6.0, "B07":6.0, "C07":6.0, "D07":6.0,
          "B08":7.0, "C08":7.0, "D08":7.0}

stains=['pSmad1/5_nc_ratio', 'betaCatenin_nuc', 'MapK_nc_ratio'] #, 'pSmad2/3_nc_ratio'
 
def IQR(data, col):
    quant_1= np.quantile(data[col], 0.25)
    quant_3 = np.quantile(data[col], 0.75)
    return(quant_3-quant_1)

def z_score_med(data, col):
    iqr=IQR(data, col)
    data[col]=(data[col]-np.median(data[col]))/iqr
    
    
n_bins_per_ax=8
n_bins=n_bins_per_ax**2


labels = list(range(1, n_bins+1))
ax_labels=range(1, n_bins_per_ax+1)
theta_bins=np.linspace(0, 1, n_bins_per_ax+1, endpoint=True)
theta_labs=theta_bins
phi_bins_abs=np.linspace(0, math.pi, n_bins_per_ax+1)
phi_labs_abs=360*phi_bins_abs/(2*math.pi)
phi_labs_abs=phi_labs_abs[:-1]


#"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Script/df/Perfect_embryos/" 
#"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances/" 
#"/data/homes/fcurvaia/distances/" 
path_in="/data/homes/fcurvaia/distances_new/" 
#path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Neural_network/"   #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Posterior/" #"/data/homes/fcurvaia/Images/Pos_inf/"
#path_in="/data/homes/fcurvaia/distances/Perfect_embryos/"
path_out_im="/data/homes/fcurvaia/Images/Neural_network/"  

#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/distances_new/" #"/data/homes/fcurvaia/distances/"
#path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/MLP/"


#wells=["D05", "B06", "C06"] + ["B08", "C08", "D08"] +["D06", "B07", "C07", "D07"]
#wells=["B08", "C08", "D08"] +["D06", "B07", "C07", "D07"]
wells=["D06", "B07", "C07", "D07"] #6 hpf
#wells=["B08", "C08", "D08"] #7 hpf
#wells=["D05", "B06", "C06"] #5_7 hpf
#wells=["C05"] #5_3 hpf
#wells=["B05"] #4_7 hpf
#wells=["B04", "C04", "D04"] #4_3 hpf
#wells=["B03", "C03", "D03"] #3_7 hpf
#wells=["B02", "C02", "D02"] #3_3 hpf

hpf=time_emb[wells[0]]



fld=Path(path_in)
files=[]
all_df=[]
to_remove=['D06_px+1741_py-0131', 'D06_px-1055_py-0118', 'B07_px+1257_py-0474', 'C07_px+1929_py-0176', "B07_px-0202_py+0631", "C07_px+0243_py-1998"]
#to_remove=["B07_px-0202_py+0631", "C07_px+0243_py-1998", "B04_px+0114_py-1436", "B05_px-0648_py+0837", "B05_px+1541_py+1373"]
all_t_max={}

index=list(product(range(1, n_bins_per_ax+1), range(1, n_bins_per_ax+1)))

for w in wells:
    for f in fld.glob(w+"*"+"_w_dist_sph_simp.csv"):
#for f in fld.glob("*_w_dist_sph_simp.csv"):
        name=f.name.split(".")[0]
        emb=name.split("_w")[0]
        if emb not in to_remove:
            try:
                fn=path_in+emb+"_w_dist_sph_simp.csv"
                feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["theta", "cur_phi"]+stains)
                well=emb.split("_")[0]
                files.append(emb)
                feat_filt["emb"]=emb
                feat_filt["hpf"]=time_emb[well]
                for stain in stains:
                    y_max=feat_filt[stain].quantile(0.99)
                    y_min=feat_filt[stain].quantile(0.025)
                    feat_filt=feat_filt.loc[feat_filt[stain]>y_min]
                    feat_filt=feat_filt.loc[feat_filt[stain]<y_max]
                    feat_filt[stain]=zscore(feat_filt[stain])
                feat_filt["rel_theta"]=feat_filt.theta/np.max(feat_filt.theta)
                #feat_filt['theta_bin_pos'] = pd.qcut(feat_filt['rel_theta'], n_bins_per_ax, labels=False)+1
                feat_filt['theta_bin_cv'] = pd.to_numeric(pd.cut(x = feat_filt['rel_theta'], bins = theta_bins, labels = ax_labels, include_lowest = False, right=True))
                feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['cur_phi'].abs(), bins = phi_bins_abs, labels = ax_labels, include_lowest = False, right=True))
    
                feat_filt["both_bins"]=0
                for i in range(len(index)):
                    all_idx=index[i]
                    idx_phi=all_idx[0]
                    idx_theta=all_idx[1]
                    feat_filt.loc[(feat_filt.phi_bin_abs==idx_phi) & (feat_filt.theta_bin_cv==idx_theta), "both_bins"]=i+1
                
                feat_filt=feat_filt.sort_values(["phi_bin_abs", "theta_bin_cv"])
                feat_filt.reset_index(inplace=True)
                all_df.append(feat_filt)
                feat_filt=0
                del feat_filt
            except ValueError:
                pass
            
"""
hpf=7
path_in="/data/homes/fcurvaia/distances/Perfect_embryos/"
path_out_im="/data/homes/fcurvaia/Images/Perfect_embryos/Random_forest/both_bins/" 
fld=Path(path_in)
stains=["low_dorsal_margin", "high_ventral_margin", "high_margin"] #, "uniform"
for f in fld.glob("*.csv"):
    name=f.name.split(".")[0]
    emb=name.split("_")[2]
    if int(emb)<34 and int(emb)%3==0:
        files.append(emb)
        fn=path_in+"Perfect_embryo_"+emb+".csv"
        feat_filt=pd.read_csv(fn, sep=",", index_col=False, usecols=["theta", "phi_abs"]+stains)
        feat_filt["emb"]=emb
        feat_filt["rel_theta"]=feat_filt.theta/np.max(feat_filt.theta)
        #feat_filt['theta_bin_pos'] = pd.qcut(feat_filt['rel_theta'], n_bins_per_ax, labels=False)+1
        feat_filt['theta_bin_cv'] = pd.to_numeric(pd.cut(x = feat_filt['rel_theta'], bins = theta_bins, labels = ax_labels, include_lowest = False, right=True))
        feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['phi_abs'].abs(), bins = phi_bins_abs, labels = ax_labels, include_lowest = False, right=True))

        feat_filt["both_bins"]=0
        for i in range(len(index)):
            all_idx=index[i]
            idx_phi=all_idx[0]
            idx_theta=all_idx[1]
            feat_filt.loc[(feat_filt.phi_bin_abs==idx_phi) & (feat_filt.theta_bin_cv==idx_theta), "both_bins"]=i+1
        
        feat_filt=feat_filt.sort_values(["phi_bin_abs", "theta_bin_cv"])
        feat_filt.reset_index(inplace=True)
        all_df.append(feat_filt)
        feat_filt=0
        del feat_filt
    
"""
        
#del files[9]
#del files[4]
#del files[3]
#del files[1]
#files.remove("C07_px+0243_py-1998")
#files.remove('B07_px-0202_py+0631')

feat_filt=pd.concat(all_df)

"""
#plt.style.use('dark_background')
f1, axs1 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(60, 30), sharey=True)
fon_size=10
#space_size=38
plt.rcParams.update({'font.size': fon_size}) 
f1.subplots_adjust(hspace=0.125)
f1.subplots_adjust(wspace=0.15)
f1.set_dpi(300)
f2, axs2 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(60, 30), sharey=True)
f2.subplots_adjust(hspace=0.125)
f2.subplots_adjust(wspace=0.125)
f2.set_dpi(300)
f3, axs3 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(60, 30))
f3.subplots_adjust(hspace=0.125)
f3.subplots_adjust(wspace=0.125)
f3.set_dpi(300)
f4, axs4 = plt.subplots(2,5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(60, 30))
f4.subplots_adjust(hspace=0.125)
f4.subplots_adjust(wspace=0.125)
f4.set_dpi(300)
"""

fon_size=5
#space_size=38
plt.rcParams.update({'font.size': fon_size}) 

feat_filt.cur_phi=np.abs(feat_filt.cur_phi)

to_pred=["rel_theta", "cur_phi"]


    


#feat_filt["rel_theta"]=feat_filt.theta/np.max(feat_filt.theta)


#feat_filt['phi_bin_abs'] = pd.to_numeric(pd.cut(x = feat_filt['phi'].abs(), bins = phi_bins_abs, labels = labels, include_lowest = True, right=False))

"""
t_max=np.max(feat_filt.theta_bin_cv)
theta_labs=360*theta_bins/(2*pi)
theta_labs=theta_labs[:t_max]
"""
X_train, X_test, y_train, y_test = train_test_split(feat_filt[stains], feat_filt[to_pred], random_state=1)

dataset_size = len(X_train)

kl_divergence_function = (lambda q, p, _: dist.kl_divergence(q, p) / tf.cast(dataset_size, dtype=tf.float32))

tf.keras.backend.clear_session()

model_tfp = tf.keras.Sequential([
    tf.keras.Input(shape=(3,),name="basket"),
    tfp.layers.DenseFlipout(16, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu, name="dense_tfp_1"),
    tfp.layers.DenseFlipout(6, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu, name="dense_tfp_2"),
    tfp.layers.DenseFlipout(2, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.softmax, name="out_tfp_pred"),
])

learning_rate = 0.01

model_tfp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='KLD',
                  metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/phi_theta_tfp", histogram_freq=1)

history = model_tfp.fit(X_train, 
                    y_train,
                    epochs=100, 
                    verbose=0, 
                    use_multiprocessing=True, 
                    callbacks=[tensorboard_callback],
                    validation_split=0.1,
                    validation_freq=25)

y_pred=model_tfp.predict(X_test)

print(np.mean(y_pred, axis=1))

#regr = MLPRegressor(random_state=1, max_iter=500, activation="relu", solver="sgd", learning_rate="invscaling").fit(X_train, y_train)

#y_pred=regr.predict(X_test)
#print(regr.score(X_test, y_test))

fig, ax=plt.subplots()
ax.scatter(y_test.rel_theta, y_pred[:,0], s=1, marker=".")
fig.savefig(path_out_im+"rel_theta_BNN_scatter.png", bbox_inches='tight', dpi=300)

fig1, ax1=plt.subplots()
ax1.scatter(y_test.cur_phi, y_pred[:,1], s=1, marker=".")
fig1.savefig(path_out_im+"cur_phi_BNN_scatter.png", bbox_inches='tight', dpi=300)
plt.close()

"""
all_conf_mat=[]
for emb in np.unique(feat_filt.emb):
    
    train=feat_filt.loc[feat_filt.emb!=emb][stains+to_pred]

    
    test=feat_filt.loc[feat_filt.emb==emb][stains+to_pred]
    print(max(test.both_bins))
    bins_to_rm=[]
    for i in range(1, n_bins+1):
        if i not in np.unique(test.both_bins):
            test.loc[len(test)]=[0,0,0]+[i]
            bins_to_rm.append(i-1)
    
    
    test=test.sort_values("both_bins")
    test["both_bins"]=test["both_bins"].astype(int)
    
    #train, test= train_test_split(feat_filt[stains+to_pred], test_size=0.3, random_state=42) #[stains+to_pred]
    
    #train, test= train_test_split(feat_filt, test_size=0.3, random_state=42)
    
    y_train= np.squeeze(pd.concat([train.pop(x) for x in to_pred], axis=1).to_numpy())
    
    x_train=train.to_numpy()
    
    #x_train = PolynomialFeatures(degree=2, include_bias=False).fit_transform(train)
    
    y_test= np.squeeze(pd.concat([test.pop(x) for x in to_pred], axis=1))
    
    x_test=test.to_numpy()
    
    
    model = RandomForestClassifier(n_estimators = 200, random_state = 42)
    
    #x_all=pd.concat([x_train, x_test], axis=0)
    #y_all=pd.concat([y_train, y_test], axis=0)
    
    model_results=model.fit(x_train, y_train)
    y_pred=model_results.predict(x_test) 
    
    #rkf = RepeatedKFold(n_splits=5, n_repeats=10)
    #y_pred = cross_val_predict(model, x_all, y_all, cv=5, n_jobs=5) 
    
    #print(model.score(x_test, y_test))
    #path_out_im="/data/homes/fcurvaia/Images/UMAP/Common/"
    
    
    #x_test = PolynomialFeatures(degree=2, include_bias=False).fit_transform(test)
    
    #path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/Images/Lin_reg/"
    #path_out_im="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/Images/Reg/"
    #path_out_im="/data/homes/fcurvaia/Images/Random_Forest/"
    
    #axs_sub=[ax1, ax2]
    #axs_conf=[ax3, ax4]
    #, sub_ax, ax_conf 
    x_plot=y_test.to_numpy()
    y_plot=y_pred
    
    conf_mat=confusion_matrix(x_plot, y_plot, normalize="true")
    for i in bins_to_rm:
        conf_mat[i,:]=np.nan
    all_conf_mat.append(conf_mat)
    all_lengths=[]
    for i in range(conf_mat.shape[0]):
        bin_i=conf_mat[i,:]
        left=bin_i[:i].copy()
        left=left[::-1].copy()
        right=bin_i[i:].copy()
        size=max(right.shape[0], left.shape[0])
        right.resize((size))
        left.resize((size), refcheck=False)
        left=np.roll(left, 1)
        cum_sum=np.cumsum(right)+ np.cumsum(left)
        length=np.argmax(cum_sum>2/3)
        all_lengths.append(length)
    
    fig, ax=plt.subplots()
    ax.plot(labels, all_lengths)
    ax.set_xlabel("bins_label")
    ax.set_ylabel("N bins")
    #ax.set_title(im)
    fig.savefig(path_out_im+str(hpf)+"_"+emb+"_"+str(n_bins)+"_plot_random_forest.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    

    disp = ConfusionMatrixDisplay.from_predictions(x_plot, y_plot, cmap="turbo", normalize="true")
    disp.plot(ax=ax_conf)
    disp.ax_.set_title(im)

    fig, ax=plt.subplots()
    hm=ax.imshow(conf_mat, cmap="inferno") #annot=True,
    #ax.set_title(im+" "+str(hpf)+" hpf")
    ax.set_yticks(np.array(list(range(0, n_bins)))+0.5, index) #, rotation=90 #bins_coord_ticks
    ax.set_xticks(np.array(list(range(0, n_bins)))+0.5, index, rotation=90) #bins_coord_ticks
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.colorbar(hm, ax=ax, label="Classification pctage")
    fig.savefig(path_out_im+str(hpf)+"_"+emb+"_"+str(n_bins)+"_conf_mat_random_forest.png", bbox_inches='tight', dpi=300)
    plt.close()
    
mean_conf_mat=np.nanmean(np.array(all_conf_mat), axis=0)
fig, ax=plt.subplots()
hm=ax.imshow(mean_conf_mat, cmap="inferno") #annot=True,
#ax.set_title(im+" "+str(hpf)+" hpf")
ax.set_yticks(np.array(list(range(0, n_bins)))+0.5, index) #, rotation=90 #bins_coord_ticks
ax.set_xticks(np.array(list(range(0, n_bins)))+0.5, index, rotation=90) #bins_coord_ticks
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
fig.colorbar(hm, ax=ax, label="Mean Classification pctage")
fig.savefig(path_out_im+str(hpf)+"_"+str(n_bins)+"_mean_conf_mat_random_forest.png", bbox_inches='tight', dpi=300)
plt.close()
        
print("Running time --- %s seconds ---\n" % (time.time() - start_time_0))

"""








