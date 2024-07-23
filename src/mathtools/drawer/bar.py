#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   bar.py
@Time    :   2024/07/20 21:02:13
@Author  :   MuliMuri
@Version :   1.0
@Desc    :   Bar drawer class
'''


import numpy as np
import matplotlib.pyplot as plt
import warnings

from .. import dataer

from typing import Any, Dict, List, Tuple


class Bar():
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

        self.bars_data = {}

        self.is_bar = False

    def bar(self,
            data: Dict[str, Dict[str, Any]]) -> 'Bar':

        if (self.is_bar):
            warnings.warn("Cannot plot multiple bar charts on the same plot")
            return self

        keys = list(data.keys())
        value_dict = list(data.values())

        for i, label in enumerate(keys):
            y = dataer.load_numpy(value_dict[i]["y_name"]) \
                if "y_data" not in value_dict[i].keys() else value_dict[i]["y_data"]

            x = np.arange(len(y)) \
                if "x_labels" not in value_dict[i].keys() else value_dict[i]["x_labels"]

            bar = self.ax.bar(x, y, label=label)

            self.bars_data[label] = bar

        self.is_bar = True

        return self

    def group_bar(self,
                  data: Dict[str, Dict[str, Any]]) -> 'Bar':

        if (self.is_bar):
            warnings.warn("Cannot plot multiple bar charts on the same plot")
            return self

        keys = list(data.keys())
        value_dict = list(data.values())

        num_datas = len(keys)

        width = 1 / (num_datas + 1)

        max_x_length = 0
        max_x_labels = None

        for i, label in enumerate(keys):
            y = dataer.load_numpy(value_dict[i]["y_name"]) \
                if "y_data" not in value_dict[i].keys() else value_dict[i]["y_data"]

            x_labels = value_dict[i]["x_labels"]
            x = np.arange(len(x_labels))

            if (max_x_length < len(x_labels)):
                max_x_length = len(x_labels)
                max_x_labels = x_labels

            delta = i - (num_datas - 1) / 2

            bar = self.ax.bar(x + delta * width, y, width, label=label)

            self.bars_data[label] = bar

        self.ax.set_xticks(np.arange(max_x_length), labels=max_x_labels)

        self.is_bar = True

        return self

    def with_top_label(self,
                       labels: List[str]) -> 'Bar':

        if (not self.is_bar):
            warnings.warn("Please plot the bar chart first")
            return self

        for i, label in enumerate(labels):
            self.ax.bar_label(self.bars_data[label], padding=3)

        return self

    def show(self) -> None:
        plt.legend()
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.tight_layout()
        plt.show()
