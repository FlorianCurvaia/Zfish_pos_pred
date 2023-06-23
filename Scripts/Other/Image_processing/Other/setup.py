#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:54:49 2023

@author: floriancurvaia
"""
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Neighbours matrix',
    ext_modules=cythonize("neigh_mat.pyx"),
    zip_safe=False,
)