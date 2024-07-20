#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   mat_window.py
@Time    :   2024/07/20 04:35:14
@Author  :   MuliMuri 
@Version :   1.0
@Desc    :   Some useful functions for numpy matrix when it needs to be windowed
'''

import numpy as np

def rolling_window(mat:np.ndarray, window_size:int) -> np.ndarray:
    """To solve ndarray has no pandas rolling function

    Args:
        mat (np.ndarray): Numpy matrix
        window_size (int): Window size

    Returns:
        np.ndarray: Windowed 2D matrix
    """
    shape = mat.shape[:-1] + (mat.shape[-1] - window_size + 1, window_size)
    strides = mat.strides + (mat.strides[-1],)

    return np.lib.stride_tricks.as_strided(mat, shape=shape, strides=strides)
