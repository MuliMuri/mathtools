#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   heatmap.py
@Time    :   2024/08/24 01:31:35
@Author  :   MuliMuri
@Version :   1.0
@Desc    :   The heatmap drawer.
'''


import numpy as np
import matplotlib.pyplot as plt
import warnings

from .. import dataer

from typing import Any, Dict, Tuple


class Heatmap():
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

        self.heatmap_datas = {}

        self.is_heatmap = False

    def imshow(self,
               data: Dict[str, Dict[str, Any]],
               rotation_deg: int = 45) -> 'Heatmap':

        if (self.is_heatmap):
            warnings.warn("Cannot plot multiple heatmaps on the same plot")
            return self

        keys = list(data.keys())
        value_dict = list(data.values())

        if (len(keys) != 1):
            raise ValueError("Only one heatmap can be plotted at a time")

        heat_data = dataer.load_numpy(value_dict[0]["heat_name"]) \
            if "heat_data" not in value_dict[0].keys() else value_dict[0]["heat_data"]

        x_ticks = value_dict[0]["x_ticks"] \
            if "x_ticks" in value_dict[0].keys() else np.arange(heat_data.shape[1])

        y_ticks = value_dict[0]["y_ticks"] \
            if "y_ticks" in value_dict[0].keys() else np.arange(heat_data.shape[0])

        if (len(x_ticks) != heat_data.shape[1] or len(y_ticks) != heat_data.shape[0]):
            raise ValueError("The length of x_ticks and y_ticks must match the shape of `heat_data`")

        self.heatmap_datas[keys[0]] = {
            "heat_data": heat_data,
            "x_ticks": x_ticks,
            "y_ticks": y_ticks,
            "plot": self.ax.imshow(heat_data, cmap="coolwarm")
        }

        plt.xticks(np.arange(len(x_ticks)), labels=x_ticks,
                   rotation=rotation_deg, ha='right', rotation_mode='anchor')

        plt.yticks(np.arange(len(y_ticks)), labels=y_ticks)

        self.is_heatmap = True

        return self

    def contourf(self,
                 data: Dict[str, Dict[str, Any]],
                 rotation_deg: int = 45) -> 'Heatmap':

        if (self.is_heatmap):
            warnings.warn("Cannot plot multiple heatmaps on the same plot")
            return self

        keys = list(data.keys())
        value_dict = list(data.values())

        if (len(keys) != 1):
            raise ValueError("Only one heatmap can be plotted at a time")

        heat_data = dataer.load_numpy(value_dict[0]["heat_name"]) \
            if "heat_data" not in value_dict[0].keys() else value_dict[0]["heat_data"]

        x_ticks = value_dict[0]["x_ticks"] \
            if "x_ticks" in value_dict[0].keys() else np.arange(heat_data.shape[1])

        y_ticks = value_dict[0]["y_ticks"] \
            if "y_ticks" in value_dict[0].keys() else np.arange(heat_data.shape[0])

        if (len(x_ticks) != heat_data.shape[1] or len(y_ticks) != heat_data.shape[0]):
            raise ValueError("The length of x_ticks and y_ticks must match the shape of `heat_data`")

        self.heatmap_datas[keys[0]] = {
            "heat_data": heat_data,
            "x_ticks": x_ticks,
            "y_ticks": y_ticks,
            "plot": self.ax.contourf(heat_data, cmap="coolwarm")
        }

        plt.xticks(np.arange(len(x_ticks)), labels=x_ticks,
                   rotation=rotation_deg, ha='right', rotation_mode='anchor')

        plt.yticks(np.arange(len(y_ticks)), labels=y_ticks)

        self.is_heatmap = True

        return self

    def with_colorbar(self,
                      bar_title: str = "") -> 'Heatmap':

        self.fig.colorbar(self.heatmap_datas[list(self.heatmap_datas.keys())[0]]['plot'], label=bar_title)

        return self

    def with_specific_text(self,
                           text_color: str = "black") -> 'Heatmap':

        for key in self.heatmap_datas.keys():
            data = self.heatmap_datas[key]

            for i in range(data["heat_data"].shape[0]):
                for j in range(data["heat_data"].shape[1]):
                    self.ax.text(j, i, f"{data['heat_data'][i, j]:.2f}",
                                 ha="center", va="center", color=text_color)

        return self

    def show(self) -> None:     # pragma: no cover
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        plt.tight_layout()
        plt.show()
