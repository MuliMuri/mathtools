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
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.interpolate import make_interp_spline
from scipy.integrate import simps

from .. import dataer
from ..utils.mat_window import rolling_window

class Plot:
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot(dict_data:dict[str, dict[str]], title:str, x_label:str, y_label:str, figsize:tuple=(12,10), is_show:bool=True) -> None:
        labels = list(dict_data.keys())
        value_dict = list(dict_data.values())

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        for i, label in enumerate(labels):
            x_range = value_dict[i]["x_range"] if "x_data" not in value_dict[i].keys() else (value_dict[i]["x_data"].min(), value_dict[i]["x_data"].max())

            y:np.ndarray = dataer.load_numpy(value_dict[i]["y_name"]) if "y_data" not in value_dict[i].keys() else value_dict[i]["y_data"]
            x:np.ndarray = np.linspace(x_range[0], x_range[1], len(y)) if "x_data" not in value_dict[i].keys() else value_dict[i]["x_data"] 

            ax.plot(x, y, label=label, linewidth=1.3)
            
        if (is_show):
            plt.legend()
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.tight_layout()
            plt.show()
        else:
            return fig, ax

    @staticmethod
    def spline(dict_data:dict[str, dict[str]], title:str, x_label:str, y_label:str, figsize:tuple=(12,10), spline_order:int=3, is_show:bool=True) -> None:
        labels = list(dict_data.keys())
        value_dict = list(dict_data.values())

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        x_splines = []
        y_splines = []

        for i, label in enumerate(labels):
            x_range = value_dict[i]["x_range"] if "x_data" not in value_dict[i].keys() else (value_dict[i]["x_data"].min(), value_dict[i]["x_data"].max())

            y:np.ndarray = dataer.load_numpy(value_dict[i]["y_name"]) if "y_data" not in value_dict[i].keys() else value_dict[i]["y_data"]
            x:np.ndarray = np.linspace(x_range[0], x_range[1], len(y)) if "x_data" not in value_dict[i].keys() else value_dict[i]["x_data"] 

            ax.scatter(x, y)
            spl = make_interp_spline(x, y, k=spline_order)
            x_spline = np.linspace(x.min(), x.max(), 500)
            y_spline = spl(x_spline)

            ax.plot(x_spline, y_spline, label=label)

            if (not is_show):
                x_splines.append(x_spline)
                y_splines.append(y_spline)
            
        if (is_show):
            plt.legend()
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.tight_layout()
            plt.show()
        else:
            return fig, ax, x_splines, y_splines

    @staticmethod
    def with_error(dict_data:dict[str, dict[str]], title:str, x_label:str, y_label:str, figsize:tuple=(12,10), ref_times:int=3, is_show:bool=True, fig:plt.Figure=None, ax:plt.Axes=None) -> None:
        labels = list(dict_data.keys())
        value_dict = list(dict_data.values())

        if (fig is None or ax is None):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        for i, label in enumerate(labels):
            x_range = value_dict[i]["x_range"] if "x_data" not in value_dict[i].keys() else (value_dict[i]["x_data"].min(), value_dict[i]["x_data"].max())

            y:np.ndarray = dataer.load_numpy(value_dict[i]["y_name"]) if "y_data" not in value_dict[i].keys() else value_dict[i]["y_data"]
            x:np.ndarray = np.linspace(x_range[0], x_range[1], len(y)) if "x_data" not in value_dict[i].keys() else value_dict[i]["x_data"] 

            std_values = np.zeros_like(y)
            std_values[:ref_times - 1] = 0
            std_values[ref_times - 1:] = rolling_window(y, ref_times).std(axis=-1, ddof=1)

            ax.fill_between(x, 
                             y - ref_times * std_values,
                             y + ref_times * std_values,
                             alpha=0.2)
            
        if (is_show):
            plt.legend()
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.tight_layout()
            plt.show()
        else:
            return fig, ax

    @staticmethod
    def with_local_zoom(dict_data:dict[str, dict[str]], title:str, x_label:str, y_label:str, figsize:tuple=(12,10), zoom_range_ratio:tuple=(0.55, 0.65), zoom_window_pos:tuple=(0.1, 0.1, 0.2, 0.2), is_show:bool=True, fig:plt.Figure=None, ax:plt.Axes=None) -> None:
        labels = list(dict_data.keys())
        value_dict = list(dict_data.values())

        if (fig is None or ax is None):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        childAx = ax.inset_axes(zoom_window_pos)
        ax.set_ylim(-5, 5)
        zoom_ylims = np.zeros((len(labels), 2))
        for i, label in enumerate(labels):
            x_range = value_dict[i]["x_range"] if "x_data" not in value_dict[i].keys() else (value_dict[i]["x_data"].min(), value_dict[i]["x_data"].max())

            y:np.ndarray = dataer.load_numpy(value_dict[i]["y_name"]) if "y_data" not in value_dict[i].keys() else value_dict[i]["y_data"]
            x:np.ndarray = np.linspace(x_range[0], x_range[1], len(y)) if "x_data" not in value_dict[i].keys() else value_dict[i]["x_data"] 

            childAx.plot(x, y, label=label, linewidth=1.3)

            childAx.set_xlim(int(x_range[1]*zoom_range_ratio[0]),int(x_range[1]*zoom_range_ratio[1]))

            zoom_y = y[int(y.shape[0]*zoom_range_ratio[0]):int(y.shape[0]*zoom_range_ratio[1])]
            zoom_ylims[i] = [zoom_y.min()-zoom_y.mean(), zoom_y.max()+zoom_y.mean()]

            mark_inset(ax, childAx, loc1=3, loc2=4, fc="none", ec='k', lw=1)
            
        childAx.set_ylim(zoom_ylims.min(), zoom_ylims.max())

        if (is_show):
            plt.legend()
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.tight_layout()
            plt.show()
        else:
            return fig, ax

    @staticmethod
    def with_density(dict_data:dict[str, dict[str]], title:str, x_label:str, y_label:str, figsize:tuple=(12,10), density_range:tuple=(0.25, 0.75), is_show:bool=True, fig:plt.Figure=None, ax:plt.Axes=None) -> None:
        labels = list(dict_data.keys())
        value_dict = list(dict_data.values())

        if (fig is None or ax is None):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        
        density_ratios = np.zeros(len(labels))

        for i, label in enumerate(labels):
            x_range = value_dict[i]["x_range"] if "x_data" not in value_dict[i].keys() else (value_dict[i]["x_data"].min(), value_dict[i]["x_data"].max())

            y:np.ndarray = dataer.load_numpy(value_dict[i]["y_name"]) if "y_data" not in value_dict[i].keys() else value_dict[i]["y_data"]
            x:np.ndarray = np.linspace(x_range[0], x_range[1], len(y)) if "x_data" not in value_dict[i].keys() else value_dict[i]["x_data"]

            ax.fill_between(x, y, where=(x >= x_range[1]*density_range[0]) & (x <= x_range[1]*density_range[1]), alpha=0.2)
            
            simps_area = simps(y, x)
            
            x_density = x[(x > x_range[1]*density_range[0]) & (x < x_range[1]*density_range[1])]
            y_density = y[(x > x_range[1]*density_range[0]) & (x < x_range[1]*density_range[1])]
            
            simps_density = simps(y_density, x_density)
            density_ratios[i] = simps_density/simps_area

            ax.text(x_range[1]*density_range[0], y.max(), f"Density: {density_ratios[i]:.2f}", fontsize=12)
            
        if (is_show):
            plt.legend()
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.tight_layout()
            plt.show()
        else:
            return fig, ax
