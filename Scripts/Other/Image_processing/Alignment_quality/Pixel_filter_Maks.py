#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:20:13 2023

@author: floriancurvaia
"""
import h5py
import numpy as np
import pandas as pd
from abbott.h5_files import *
import time
#from functools import reduce
#import itertools
#import numba
import argparse
from pathlib import Path
#import napari
from skimage.measure import regionprops_table
from tqdm import tqdm
#import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.stats import pearsonr
import nest_asyncio
nest_asyncio.apply()



start_time_0=time.time()
CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--image",  # name on the CLI - drop the `--` for positional/required parameters
  nargs=1,  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)


args = CLI.parse_args()
path_in_dist='/data/homes/fcurvaia/features/' #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/work/fcurvaia/"#'/data/homes/fcurvaia/features/'
path_out_dist="/data/homes/fcurvaia/distances/" #"/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/"#/data/homes/fcurvaia/distances/"
#path_in="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/"#"/data/active/fcurvaia/Segmented_files/"#'/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/'#B08_px+1076_py-0189.h5'
#fn='/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B07_px+1257_py-0474.h5'
path_in='/data/active/sshami/20220716_experiment3_aligned/'

im=args.image[0] #"B08_px+1076_py-0189"#str(sys.argv[1])
print(im)
fn=path_in+im+".h5"
"""
im2=args.image[1]
print(im2)

fn2=path_in+im2+".h5"
#fn_dist=path_in_dist+im+".csv"
"""
"""
with h5py.File(fn) as f:
    ref_dapi = to_numpy(h5_select(f, attr_select={'stain': 'DAPI', 'cycle': 1, 'level': 1})[0])
    test_dapi = to_numpy(h5_select(f, attr_select={'stain': 'DAPI', 'cycle': 0, 'level': 1})[0])
    nucs = to_numpy(h5_select(f, attr_select={'stain': 'nucleiRaw'})[0])
"""
"""
with h5py.File(fn2) as f:
    for dset in h5_select(f, {'stain': 'DAPI', "cycle":0, "level":1}):
        test_dapi = to_numpy(h5_select(f, attr_select={'stain': 'DAPI', 'cycle': 0, 'level': 1})[0])
"""


#labels = np.unique(nucs)[1:]
#print(len(labels))

print("load files and argument --- %s seconds ---\n" % (time.time() - start_time_0))



def label_wise_(lbls, ref_dapi, test_dapi):
    corrs = []

    props = regionprops_table(lbls, properties=('label', 'slice'))
    labels = props['label']
    slices = props['slice']

    for slc in tqdm(slices):
        nuc = lbls[slc]
        ref = ref_dapi[slc]
        test = test_dapi[slc]
        ref_reg = ref[np.where(nuc>0)]
        test_reg = test[np.where(nuc>0)]
        r, p = pearsonr(ref_reg, test_reg)
        corrs.append(round(r, 6))

    return pd.DataFrame({'Label': labels, 'corrs': corrs}).set_index('Label')['corrs']


def _per_cycle(fn, ref_cycle=1, stain='DAPI', label='nucleiRaw'):
    fn = Path(fn)
    with h5py.File(fn) as f:
        lbls_dset = h5_select(f, {'stain': label})[0]
        level = lbls_dset.attrs['level']
        ref_dset = h5_select(f, {'stain': stain, 'cycle': ref_cycle, 'level': level})[0]

        lbls = to_numpy(lbls_dset)
        ref = to_numpy(ref_dset)

        corrs = {}
        for test_dset in h5_select(f, {'stain': stain, 'level': level}):
            test = to_numpy(test_dset)
            corrs[int(test_dset.attrs['cycle'])] = label_wise_(lbls, ref, test)

        corrs = pd.DataFrame(corrs)
        corrs['filename_prefix'] = fn.stem
        corrs = corrs.reset_index().set_index(['filename_prefix', 'Label'])
        return corrs

"""
def calculate__per_cycle(fld, out_dir='_nR3'):
    fld = Path(fld)
    for fn in fld.glob('*.h5'):
        print(fn)
        corrs = _per_cycle(fn)
        out_path = fld / out_dir / f'{fn.stem}_.csv'
        out_path.parent.mkdir(exist_ok=True)
        corrs.to_csv(out_path)

"""
start_time_1=time.time()

#corr_df=label_wise_(nucs, ref_dapi, test_dapi)

corr_df=_per_cycle(fn)

corr_df.to_csv(path_out_dist+im+"_corr_dapi.csv", sep=",")


print("find correlations --- %s seconds ---\n" % (time.time() - start_time_1))





