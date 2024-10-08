#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   plot.py
@Time    :   2024/07/18 21:17:47
@Author  :   MuliMuri
@Version :   1.0
@Desc    :   Plot drawer class
'''


import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.interpolate import make_interp_spline
from scipy.integrate import simpson

from .. import dataer
from ..utils.mat_window import rolling_window

from typing import Any, Dict, List, Tuple


class Plot():
    def __init__(self,
                 title: str,
                 x_label: str,
                 y_label: str,
                 figsize: Tuple[int, int] = (12, 10)) -> None:
        """Plot drawer class

        Args:
            title (str): Title of the figure
            x_label (str): X-axis label
            y_label (str): Y-axis label
            figsize (Tuple[int, int], optional): To appoint the figure size. Defaults to (12, 10).
        """

        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        self.childAxs = []

        self.data = {}

        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.draw_num = 0

    def plot(self,
             data: Dict[str, Dict[str, Any]]) -> 'Plot':
        """Basic function

        Args:
            data (Dict[str, Dict[str, Any]]): Drawing data

        Returns:
            Plot: Return self plot object for chaining
        """

        keys1 = list(data.keys())
        keys2 = list(self.data.keys())

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

            color_index = self.draw_num % len(self.color_cycle)

            self.ax.plot(x, y, label=label, linewidth=1.3, color=self.color_cycle[color_index])

            data[label]['color_index'] = color_index

        data.update(self.data)
        self.data = data

        self.draw_num += 1

        return self

    def spline(self,
               data: Dict[str, Dict[str, Any]],
               spline_order: int = 3) -> 'Plot':
        """Basic function

        Args:
            data (Dict[str, Dict[str, Any]]): Drawing data
            spline_order (int, optional): Spline fitting power. Defaults to 3.

        Returns:
            Plot: Return self plot object for chaining
        """

        keys1 = list(data.keys())
        keys2 = list(self.data.keys())

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

            color_index = self.draw_num % len(self.color_cycle)

            self.ax.scatter(x, y, color=self.color_cycle[color_index])
            spl = make_interp_spline(x, y, k=spline_order)
            x_spline = np.linspace(x.min(), x.max(), 500)
            y_spline = spl(x_spline)

            self.ax.plot(x_spline, y_spline, label=label, linewidth=1.3, color=self.color_cycle[color_index])

            data[label]['color_index'] = color_index

            data[label]['x_data'] = x_spline
            data[label]['y_data'] = y_spline

        data.update(self.data)
        self.data = data

        self.draw_num += 1

        return self

    def with_error_std(self,
                       labels: List[str],
                       ref_times: int = 3) -> 'Plot':
        """Stytle function

        Args:
            labels (List[str]): Appoint the labels to add error range by padding
            ref_times (int, optional): Error reference time step. Defaults to 3.

        Returns:
            Plot: Return self plot object for chaining
        """

        value_dict = [self.data[label] for label in labels]

        for i, label in enumerate(labels):
            x_range = value_dict[i]["x_range"] \
                if "x_data" not in value_dict[i].keys() else (value_dict[i]["x_data"].min(), value_dict[i]["x_data"].max())

            y = dataer.load_numpy(value_dict[i]["y_name"]) \
                if "y_data" not in value_dict[i].keys() else value_dict[i]["y_data"]

            x = np.linspace(x_range[0], x_range[1], len(y)) \
                if "x_data" not in value_dict[i].keys() else value_dict[i]["x_data"]

            std_values = np.zeros_like(y)
            std_values[:~ref_times - 1] = 0
            std_values[ref_times - 1:] = rolling_window(y, ref_times).std(axis=-1, ddof=1)

            self.ax.fill_between(x,
                                 y - ref_times * std_values,
                                 y + ref_times * std_values,
                                 alpha=0.2,
                                 color=self.color_cycle[value_dict[i]['color_index']])

        return self

    def with_local_zoom(self,
                        labels: List[str],
                        zoom_range_ratio: Tuple[float, float] = (0.55, 0.65),
                        zoom_window_pos: Tuple[float, ...] = (0.1, 0.1, 0.2, 0.2),
                        is_zoom_ylims: bool = False) -> 'Plot':
        """Stytle function

        Args:
            labels (List[str]): Appoint the labels to add zoom window
            zoom_range_ratio (Tuple[float, float], optional): The location where the data needs to be enlarged. \
                The range is [0,1]. Defaults to (0.55, 0.65).

            zoom_window_pos (Tuple[float, ...], optional): The child window position. Defaults to (0.1, 0.1, 0.2, 0.2).
            is_zoom_ylims (bool, optional): Is need to zoom ylims. Defaults to False.

        Returns:
            Plot: Return self plot object for chaining
        """

        value_dict = [self.data[label] for label in labels]

        childAx = self.ax.inset_axes(zoom_window_pos)
        if (is_zoom_ylims):
            zoom_ylims = np.zeros((len(labels), 2))

        for i, label in enumerate(labels):
            x_range = value_dict[i]["x_range"] \
                if "x_data" not in value_dict[i].keys() else (value_dict[i]["x_data"].min(), value_dict[i]["x_data"].max())

            y = dataer.load_numpy(value_dict[i]["y_name"]) \
                if "y_data" not in value_dict[i].keys() else value_dict[i]["y_data"]

            x = np.linspace(x_range[0], x_range[1], len(y)) \
                if "x_data" not in value_dict[i].keys() else value_dict[i]["x_data"]

            childAx.plot(x, y, label=label, linewidth=1.3, color=self.color_cycle[value_dict[i]['color_index']])

            childAx.set_xlim(int(x_range[1] * zoom_range_ratio[0]), int(x_range[1] * zoom_range_ratio[1]))

            if (is_zoom_ylims):
                zoom_y = y[int(y.shape[0] * zoom_range_ratio[0]):int(y.shape[0] * zoom_range_ratio[1])]
                zoom_ylims[i] = [zoom_y.min() - zoom_y.mean(), zoom_y.max() + zoom_y.mean()]

        mark_inset(self.ax, childAx, loc1=3, loc2=4, fc="none", ec='k', lw=1)
        if (is_zoom_ylims):
            self.ax.set_ylim(zoom_ylims.min(), zoom_ylims.max())
            childAx.set_ylim(zoom_ylims.min(), zoom_ylims.max())

        self.childAxs.append(childAx)

        return self

    def with_density(self,
                     labels: List[str],
                     density_range: tuple = (0.25, 0.75)) -> 'Plot':
        value_dict = [self.data[label] for label in labels]

        density_ratios = np.zeros(len(labels))

        for i, label in enumerate(labels):
            x_range = value_dict[i]["x_range"] \
                if "x_data" not in value_dict[i].keys() else (value_dict[i]["x_data"].min(), value_dict[i]["x_data"].max())

            y = dataer.load_numpy(value_dict[i]["y_name"]) \
                if "y_data" not in value_dict[i].keys() else value_dict[i]["y_data"]

            x = np.linspace(x_range[0], x_range[1], len(y)) \
                if "x_data" not in value_dict[i].keys() else value_dict[i]["x_data"]

            self. ax.fill_between(x, y,
                                  where=(x >= x_range[1] * density_range[0]) & (x <= x_range[1] * density_range[1]),
                                  alpha=0.2,
                                  color=self.color_cycle[value_dict[i]['color_index']])

            simps_area = simpson(y, x=x)

            x_density = x[(x > x_range[1] * density_range[0]) & (x < x_range[1] * density_range[1])]
            y_density = y[(x > x_range[1] * density_range[0]) & (x < x_range[1] * density_range[1])]

            simps_density = simpson(y_density, x=x_density)
            density_ratios[i] = simps_density / simps_area

            self.ax.text(x_range[1] * density_range[0], y.max(),
                         f"Density: {density_ratios[i]:.2f}",
                         fontsize=12,
                         color=self.color_cycle[value_dict[i]['color_index']])

        return self

    def show(self) -> None:     # pragma: no cover
        plt.legend()
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.tight_layout()
        plt.show()
