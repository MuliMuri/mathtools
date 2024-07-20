#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   globals.py
@Time    :   2024/07/18 21:44:47
@Author  :   MuliMuri 
@Version :   1.0
@Desc    :   Global variables
'''

from .db.data_handler import DataHandler

class Globals:
    _dataer:DataHandler = None

    @property
    def dataer(self) -> DataHandler:
        return self._dataer
    
    @dataer.setter
    def dataer(self, value:DataHandler) -> None:
        self._dataer = value

_globals = Globals()
dataer = _globals.dataer
