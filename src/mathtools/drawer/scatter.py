#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   scatter.py
@Time    :   2024/08/28 01:12:50
@Author  :   MuliMuri
@Version :   1.0
@Desc    :   Scatter drawer class
'''


import numpy as np
import matplotlib.pyplot as plt
import warnings

from .. import dataer

from typing import Any, Dict, List, Tuple
from matplotlib.colors import rgb2hex, hex2color


class Scatter():
    def __init__(self,
                 title: str,
                 x_label: str,
                 y_label: str,
                 figsize: Tuple[int, int] = (12, 10)) -> None:

        self.title = title
        self.x_label = x_label
        self.y_label = y_label

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)

        self.scattered_data = {}

        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.draw_num = 0

    def scatter(self,
                data: Dict[str, Dict[str, Any]]) -> 'Scatter':

        keys1 = list(data.keys())
        keys2 = list(self.scattered_data.keys())

        diff_keys = list(set(keys1) - set(keys2))
        value_dict = [data[key] for key in diff_keys]

        for i, label in enumerate(diff_keys):
            y = dataer.load_numpy(value_dict[i]["y_name"]) \
                if "y_data" not in value_dict[i].keys() else value_dict[i]["y_data"]

            value_dict[i]["x_range"] = (0, len(y)) \
                if "x_range" not in value_dict[i].keys() else value_dict[i]["x_range"]

            x_range = value_dict[i]["x_range"] \
                if "x_data" not in value_dict[i].keys() else (value_dict[i]["x_data"].min(), value_dict[i]["x_data"].max())

            x = np.linspace(x_range[0], x_range[1], len(y)) \
                if "x_data" not in value_dict[i].keys() else value_dict[i]["x_data"]

            sizes = 50 \
                if "sizes" not in value_dict[i].keys() else value_dict[i]["sizes"]

            color_index = self.draw_num % len(self.color_cycle)

            self.ax.scatter(x, y, label=label, c=self.color_cycle[color_index], s=sizes)

            data[label]['color_index'] = color_index

        data.update(self.scattered_data)
        self.scattered_data = data

        self.draw_num += 1

        return self

    def cluster(self,
                data: Dict[str, Dict[str, Any]]) -> 'Scatter':

        keys1 = list(data.keys())
        keys2 = list(self.scattered_data.keys())

        diff_keys = list(set(keys1) - set(keys2))
        value_dict = [data[key] for key in diff_keys]

        for i, label in enumerate(diff_keys):
            cluster_scattered_data = {}

            y = dataer.load_numpy(value_dict[i]["y_name"]) \
                if "y_data" not in value_dict[i].keys() else value_dict[i]["y_data"]

            x = dataer.load_numpy(value_dict[i]["x_name"]) \
                if "x_data" not in value_dict[i].keys() else value_dict[i]["x_data"]

            sizes = 50 \
                if "sizes" not in value_dict[i].keys() else value_dict[i]["sizes"]

            category_result = value_dict[i]["category_result"]

            clusters_num = np.max(category_result) - np.min(category_result) + 1

            color_pool = tuple([(np.random.random(), np.random.random(), np.random.random()) for i in range(clusters_num)])
            color_pool = np.array([rgb2hex(x) for x in color_pool])

            category_names = value_dict[i]["category_names"] \
                if "category_names" in value_dict[i].keys() else None

            for category_index in range(np.min(category_result), np.max(category_result) + 1):
                self.ax.scatter(x[category_result == category_index],
                                y[category_result == category_index],
                                label=category_names[category_index],
                                c=color_pool[category_index],
                                s=sizes)

                cluster_scattered_data[category_names[category_index]] = {
                    "x_data": x[category_result == category_index],
                    "y_data": y[category_result == category_index],
                    "color": color_pool[category_index],
                    "is_cluster": True
                }

            data.pop(label)
            data.update(cluster_scattered_data)

        data.update(self.scattered_data)
        self.scattered_data = data

        self.draw_num += 1

        return self

    def with_centroid(self,
                      labels: List[str],
                      centroid_label: str = "Centroid",
                      centroid_color_amount: float = 0.2) -> 'Scatter':

        points = np.zeros((len(labels), 2))
        cs = []

        for i, label in enumerate(labels):
            if (label not in self.scattered_data.keys()) or \
                    ("is_cluster" not in self.scattered_data[label].keys()):
                warnings.warn(f"Label {label} is not exists or not cluster, skip drawing centroid.")
                continue

            x = np.mean(self.scattered_data[label]["x_data"])
            y = np.mean(self.scattered_data[label]["y_data"])
            rgb_color = np.array(hex2color(self.scattered_data[label]["color"]))

            c = rgb_color * (1 - centroid_color_amount)

            points[i] = [x, y]
            cs.append(c)

        self.ax.scatter(points[:, 0], points[:, 1], edgecolors='black', c=cs, s=100, marker='o')
        self.ax.scatter([], [], label=centroid_label, edgecolors='black', c='none', s=100, marker='o')        # To add legend

        return self

    def show(self) -> None:     # pragma: no cover
        plt.legend()
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.tight_layout()
        plt.show()
