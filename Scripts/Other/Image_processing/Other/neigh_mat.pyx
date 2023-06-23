#!python
#cython: language_level=3
#distutils: extra_compile_args=-fopenmp
#distutils: extra_link_args=-fopenmp
"""
Created on Tue Mar  7 11:33:59 2023

@author: floriancurvaia
"""

import numpy as np
cimport cython
from cython.parallel import prange


DTYPE = np.intc


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cdef int index(int[:] arr, int e) nogil:
    cdef Py_ssize_t len_arr, i
    len_arr = arr.shape[0]
    for i in range(len_arr):
        if arr[i] == e:
            return i


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def adjacency_matrix(int[:, :, :] arr):
    cdef int *index_shifts = [-1, 1]
    cdef Py_ssize_t z_max = arr.shape[0]
    cdef Py_ssize_t y_max = arr.shape[1]
    cdef Py_ssize_t x_max = arr.shape[2]
    cdef Py_ssize_t max_label = np.max(arr)
    cdef int v, vn
    cdef Py_ssize_t x, y, z, nx, ny, nz, n
    

    neighborhood_matrix = np.zeros((max_label+1, max_label+1), dtype=np.float32)
    cdef float[:, :] result_view = neighborhood_matrix

    for z in prange(z_max, nogil=True):
        for y in range(y_max):
            for x in range(x_max):
                v = arr[z, y, x]
                for n in range(2):
                    nx = x + index_shifts[n]
                    if (nx>=0) and (nx<x_max):
                        vn = arr[z, y, nx]
                        result_view[v, vn] += 1
                for n in range(2):
                    ny = y + index_shifts[n]
                    if (ny>=0) and (ny<y_max):
                        vn = arr[z, ny, x]
                        result_view[v, vn] += 1
                for n in range(2):
                    nz = z + index_shifts[n]
                    if (nz>=0) and (nz<z_max):
                        vn = arr[nz, y, x]
                        result_view[v, vn] += 1
    return neighborhood_matrix




