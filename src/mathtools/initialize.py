#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   initialize.py
@Time    :   2024/07/18 05:24:04
@Author  :   MuliMuri 
@Version :   1.0
@Desc    :   To initialize the library
'''

import matplotlib as mpl

from .globals import _globals
from .db.data_handler import DataHandler

def init(task_name:str) -> None:
    _globals.dataer = DataHandler(task_name)

def update_mpl_params(parmas:dict={
    'font.size': 18,
    'font.family': 'SimHei',
    'axes.unicode_minus': False
}) -> None:
    mpl.rcParams.update(parmas)
    
    from aquarel import load_theme
    load_theme("arctic_light").apply()

class Dataer:
    def __getattr__(self, name):
        return getattr(_globals.dataer, name)

dataer:DataHandler = Dataer()
