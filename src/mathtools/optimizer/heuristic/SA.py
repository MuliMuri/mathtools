#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   SA.py
@Time    :   2024/08/29 06:00:07
@Author  :   MuliMuri
@Version :   1.0
@Desc    :   Simulated Annealing Algorithm (SA)
'''


import numpy as np

from ...algorithm.perturbation import Perturbation


class SA():
    def __init__(self,
                 init_x: np.ndarray,
                 target_func,
                 st_func,
                 iter: int = 100,
                 T0: float = 100,
                 Tf: float = 0.01,
                 alpha: float = 0.99,
                 perturbation_func=None,
                 is_log: bool = False) -> None:

        self.target_func = target_func
        self.st_func = st_func
        self.iter = iter
        self.x = init_x
        self.T = T0
        self.Tf = Tf
        self.alpha = alpha

        self.is_log = is_log

        self.current_result = None

        self.perturbation_func = Perturbation.gaussian if perturbation_func is None else perturbation_func

        self.history_records = {'results': [], 'Ts': []}

    def gen_new_solution(self, x: np.ndarray) -> tuple:
        while True:
            x_new = self.perturbation_func(x)
            if self.st_func(x_new):
                break

        return x_new

    def metrospolis(self, result, result_new):
        if result_new < result:
            return True
        else:
            return np.random.rand() < np.exp(-(result_new - result) / self.T)

    def best(self):
        pass

    def run(self) -> np.ndarray:
        count = 0

        while self.T > self.Tf:
            result_local_best = None
            for i in range(self.iter):
                self.current_result = self.target_func(self.x)
                x_new = self.gen_new_solution(self.x)
                result_new = self.target_func(x_new)
                if (self.metrospolis(self.current_result, result_new)):
                    self.x = x_new
                    if (result_local_best is None) or (result_new < result_local_best):
                        result_local_best = result_new

                self.history_records['results'].append(result_local_best)
                self.history_records['Ts'].append(self.T)

                count += 1

            if (self.is_log and count % self.iter == 0):
                print(f'Current T: {self.T}, Current x: {self.x}, Current result: {self.target_func(self.x)}')

            self.T *= self.alpha

        return self.x, self.target_func(self.x)
