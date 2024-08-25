#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   matlab.py
@Time    :   2024/07/20 07:06:31
@Author  :   MuliMuri
@Version :   1.0
@Desc    :   Interacting with MATLAB mat
'''

import numpy as np
import scipy.io as sio

from typing import Dict


def read_mat(file_path: str) -> Dict[str, np.ndarray]:
    """Read .mat file

    Args:
        file_path (str): The path of the .mat file

    Returns:
        dict (str, np.ndarray): The data in the .mat file
    """
    mat_dict = sio.loadmat(file_path)
    mat_dict.pop("__header__")
    mat_dict.pop("__version__")
    mat_dict.pop("__globals__")

    return mat_dict


def save_mat(file_path: str, data: Dict[str, np.ndarray]) -> None:
    """Save data to .mat file

    Args:
        file_path (str): The path of the .mat file
        data (dict[str, np.ndarray]): The data to be saved
    """
    sio.savemat(file_path, data)
