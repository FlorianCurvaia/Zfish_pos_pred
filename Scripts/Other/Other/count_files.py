#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:48:28 2023

@author: floriancurvaia
"""
from pathlib import Path

import matplotlib.pyplot as plt

path_in='/data/active/sshami/20220716_experiment3_aligned/'

fld=Path(path_in)

time_emb={"B02":3.3, "C02":3.3, "D02":3.3, "B03":3.7, "C03":3.7, "D03":3.7, "B04":4.3, "C04":4.3, "D04":4.3,
          "B05":4.7, "C05":5.3, "D05":5.7, "B06":5.7, "C06":5.7, "D06":6.0, "B07":6.0, "C07":6.0, "D07":6.0,
          "B08":7.0, "C08":7.0, "D08":7.0}

files=[]
files_wells=[]
for f in fld.glob("*.h5"):
    name=f.name
    files.append(time_emb[name.split("_")[0]])
    files_wells.append(name.split("_")[0])

path_out_im="/data/homes/fcurvaia/Images/Pos_inf/"
fig, ax=plt.subplots()
ax.hist(files)
fig.savefig(path_out_im+"hpf_counts.png")
plt.close()

fig, ax=plt.subplots()
ax.hist(files_wells)
fig.savefig(path_out_im+"wells_counts.png")
plt.close()