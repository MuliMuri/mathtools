#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   perturbation.py
@Time    :   2024/08/29 07:28:25
@Author  :   MuliMuri
@Version :   1.0
@Desc    :   Avoid some perturbation functions
'''

import numpy as np


class Perturbation:
    @staticmethod
    def gaussian(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return x + sigma * np.random.normal(size=x.shape)

    @staticmethod
    def uniform_random(x: np.ndarray, delta: float = 1.0) -> np.ndarray:
        return x + np.random.uniform(-delta, delta, size=x.shape)

    @staticmethod
    def neighbor(x: np.ndarray, delta: float = 1.0) -> np.ndarray:
        return x + delta * np.random.choice([-1, 1], size=x.shape)

    @staticmethod
    def cauchy(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        return x + gamma * np.random.standard_cauchy(size=x.shape)
