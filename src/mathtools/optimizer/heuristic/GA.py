#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   GA.py
@Time    :   2024/08/29 11:51:47
@Author  :   MuliMuri
@Version :   1.0
@Desc    :   Genetic Algorithm (GA)
'''

import numpy as np


class GA():
    def __init__(self,
                 target_func,
                 meta_dim: int,
                 bounds: np.ndarray,
                 st_func=None,
                 dna_size: int = 24,
                 pop_size: int = 200,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.003,
                 generation: int = 200) -> None:

        self.target_func = target_func
        self.st_func = st_func

        self.meta_dim = meta_dim

        self.dna_size = dna_size
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generation = generation
        self.bounds = bounds

    def init_population(self) -> np.ndarray:
        # POP_SIZE, META_DIM, DNA_SIZE
        x_pop = np.random.randint(0, 2, size=(self.pop_size, self.meta_dim, self.dna_size))
        for i in range(self.pop_size):
            if self.st_func is not None:
                while not self.st_func(self.translate_DNA(x_pop[i])):
                    x_pop[i] = np.random.randint(0, 2, size=(self.meta_dim, self.dna_size))

        return x_pop

    def get_fitness(self, x_pop) -> float:
        x_reals = self.translate_DNA(x_pop).T
        ys = self.target_func(x_reals)

        return -(ys - np.max(ys)) + 1e-3

    def translate_DNA(self, x_pop: np.ndarray) -> np.ndarray:
        return x_pop.dot(2 ** np.arange(self.dna_size)[::-1])                   \
            / float(2 ** self.dna_size - 1)                                     \
            * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]

    def crossover_and_mutation(self, x_pop: np.ndarray) -> np.ndarray:
        new_pop = []
        for _, father in enumerate(x_pop):
            child = father.copy()
            if np.random.rand() < self.crossover_rate:
                mother = x_pop[np.random.randint(0, self.pop_size)].copy()     # Randomly pick another as mother

                cross_points = np.random.randint(0, self.dna_size, size=self.meta_dim)

                for i, cross_point in enumerate(cross_points):
                    child[i, cross_point:] = mother[i, cross_point:]

                child = self.mutation(child)

                if self.st_func is not None and not self.st_func(self.translate_DNA(child).T):
                    continue

                new_pop.append(child)

        return np.concatenate((x_pop, np.array(new_pop)), axis=0)

    def mutation(self, x_binary) -> np.ndarray:
        if np.random.rand() < self.mutation_rate:
            mutate_dna = np.random.randint(0, 2, self.meta_dim).astype(bool)
            mutate_points = np.random.randint(0, self.dna_size, mutate_dna.sum())

            dna = x_binary[mutate_dna]
            for i, mutate_point in enumerate(mutate_points):
                dna[i, mutate_point] ^= 1

            x_binary[mutate_dna] = dna

        return x_binary

    def select(self, x_pop, fitness) -> np.ndarray:
        idx = np.random.choice(np.arange(x_pop.shape[0]), size=self.pop_size, replace=True, p=fitness / fitness.sum())

        return x_pop[idx]

    def run(self) -> np.ndarray:
        x_pop = self.init_population()
        for iter in range(self.generation):

            next_pop = self.crossover_and_mutation(x_pop)

            x_pop = self.select(next_pop, self.get_fitness(next_pop))

        fitness = self.get_fitness(x_pop)
        max_fitness_idx = np.argmax(fitness)

        # print("max_fitness:", fitness[max_fitness_idx])
        x_reals = self.translate_DNA(x_pop)

        return x_reals[max_fitness_idx], self.target_func(x_reals[max_fitness_idx])

        # print("最优的基因型：", x_pop[max_fitness_idx])
        # print("x: ", x_reals[max_fitness_idx])
