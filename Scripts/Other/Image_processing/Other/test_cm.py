#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:48:27 2023

@author: floriancurvaia
"""
#import sys
import h5py
import numpy as np
#import math
from abbott.h5_files import *
import time
import pandas as pd
import numba
import argparse

#fn="/Users/floriancurvaia/Desktop/Uni/ETH/Labos/Pelkmans/active/fcurvaia/Segmented_files/B08_px+1076_py-0189.h5"

#h5_summary(fn)

start_time_0=time.time()
CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--image",  # name on the CLI - drop the `--` for positional/required parameters
  nargs=1,  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)
CLI.add_argument(
  "--colnames",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)

args = CLI.parse_args()
path_in_dist="/data/homes/fcurvaia/distances/"

path_in='/data/active/sshami/20220716_experiment3_aligned/'
#path_in="/data/active/fcurvaia/Segmented_files/"
im=args.image[0]
fn=path_in+im+".h5"
print(args.image)
print(args.colnames)