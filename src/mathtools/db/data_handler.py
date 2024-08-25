#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   data_handler.py
@Time    :   2024/07/18 05:34:32
@Author  :   MuliMuri
@Version :   1.0
@Desc    :   To read/write packaged data from/into the database
'''

import copy
import io
import numpy as np
import pickle
import sqlite3
import warnings

from typing import Dict, List

from .sqlite import SQLite


def singleton(cls):
    _instance = {}

    def inner(task_name):
        if (cls, task_name) not in _instance:
            _instance[(cls, task_name)] = cls(task_name)

        return _instance[(cls, task_name)]
    return inner


class ObservableDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observers = []

    def add_observer(self, observer_func):
        self._observers.append(observer_func)

    def _notify_observers(self, key, old_value, new_value):
        for observer in self._observers:
            observer(self, key, old_value, new_value)

    def __setitem__(self, key, value):
        old_value = self.get(key, None)
        super().__setitem__(key, value)
        if old_value != value:
            self._notify_observers(key, old_value, value)


@singleton
class DataHandler():
    def __init__(self, task_name: str) -> None:
        self.task_name = task_name

        self.db = self.__init_database()

        self.__init_table()

        self.config = self.__init_config()
        self.config.add_observer(self.__update_config)

    def __init_database(self) -> 'SQLite':
        if (".db" != self.task_name[-3:]):
            db = SQLite(f"{self.task_name}.db")
        else:
            db = SQLite(self.task_name)

        return db

    def __init_table(self):
        # ndarray table
        if (self.db.is_table_exists("ndarray") is False):
            self.db.create_table("ndarray", ["id", "data", "shape"])

        # object table
        if (self.db.is_table_exists("object") is False):
            self.db.create_table("object", ["id", "data", "info"])

        # config table
        if (self.db.is_table_exists("config") is False):
            self.db.create_table("config", ["key", "value"])

    def __init_config(self) -> 'ObservableDict':
        config = ObservableDict()

        # Init data mapper
        config['data_mapper'] = {}

        result = self.db.query("config", "WHERE key='data_mapper'")[0]

        if (result.__len__() != 0):
            pkl_obj = pickle.loads(result[0][1])
            if (pkl_obj is dict):
                config['data_mapper'] = pkl_obj
        else:
            self.db.insert("config", {"key": "data_mapper", "value": pickle.dumps(config['data_mapper'])})

        # Init data id ptr
        config['data_id_ptr'] = -1

        result = self.db.query("config", "WHERE key='data_id_ptr'")[0]

        if (result.__len__() != 0):
            config['data_id_ptr'] = int(result[0][1])
        else:
            self.db.insert("config", {"key": "data_id_ptr", "value": config['data_id_ptr']})

        return config

    def __update_config(self, config: ObservableDict, key: str, old_value: object, new_value: object):
        if (key == "data_mapper"):
            out = io.BytesIO()
            pickle.dump(new_value, out)
            out.seek(0)
            data = sqlite3.Binary(out.read())

            self.db.update("config", {"value": data}, "WHERE key='data_mapper'")

        else:
            self.db.update("config", {"value": new_value}, f"WHERE key='{key}'")

    def __get_data_id(self, temp_dict: Dict, keys: List[str]) -> int:
        if (keys == ['']):
            raise KeyError("The keys is empty, can't insert the values.")

        key = keys[0]

        if (key in temp_dict):
            if (len(keys) > 1):
                return self.__get_data_id(temp_dict[key], keys[1:])

            else:
                # The last key
                return temp_dict[key]

        else:
            # Current key is not in the dict
            return -1

    def __check_keys_and_insert(self, temp_dict: Dict, keys: List[str], values: object, full_path: str, is_force):
        if (keys == ['']):
            raise KeyError("The keys is empty, can't insert the values.")

        key = keys[0]

        if (key in temp_dict):
            if (len(keys) > 1):
                return self.__check_keys_and_insert(temp_dict[key], keys[1:], values, full_path, is_force)

            else:
                # The last key
                if (key in temp_dict):
                    if (not is_force):
                        warnings.warn(f"The path {full_path} already exists. \
                            If you want to force update, please set is_force=True.")

                    return False

        else:
            # Current key is not in the dict
            if (len(keys) > 1):
                temp_dict[key] = {}
                return self.__check_keys_and_insert(temp_dict[key], keys[1:], values, full_path, is_force)
            else:
                temp_dict[key] = values
                return True

    def save_numpy(self, dict_str: str, ndarray: np.ndarray, is_force: bool = False, split_str: str = "."):
        keys = dict_str.split(split_str)

        data_mapper = copy.deepcopy(self.config['data_mapper'])
        data_id_ptr = self.config['data_id_ptr']

        # Check the key and insert the ndarray id
        if (self.__check_keys_and_insert(data_mapper, keys, data_id_ptr + 1, dict_str, is_force) is True):
            # Save the ndarray
            out = io.BytesIO()
            np.save(out, ndarray)
            out.seek(0)
            data = sqlite3.Binary(out.read())

            self.db.insert("ndarray", {
                "id": data_id_ptr + 1,
                "data": data,
                "shape": str(ndarray.shape)
            })

            # Update mapper and id_ptr
            self.config['data_mapper'] = data_mapper
            self.config['data_id_ptr'] += 1

            return True
        elif (is_force):
            # Update the ndarray
            out = io.BytesIO()
            np.save(out, ndarray)
            out.seek(0)
            data = sqlite3.Binary(out.read())

            self.db.update("ndarray", {
                "data": data,
                "shape": str(ndarray.shape)
            }, f"WHERE id={self.__get_data_id(data_mapper, keys)}")

            return True

    def load_numpy(self, dict_str: str, split_str: str = ".") -> 'np.ndarray':
        keys = dict_str.split(split_str)

        data_id = self.__get_data_id(self.config['data_mapper'], keys)

        if (data_id == -1):
            raise KeyError(f"The path {dict_str} is not exists.")

        result = self.db.query("ndarray", f"WHERE id={data_id}")[0]
        assert len(result) == 1

        out = io.BytesIO()
        out.write(result[0][1])
        out.seek(0)
        return np.load(out)

    def save_obj(self, dict_str: str, obj: object, info: str = "", split_str: str = ".", is_force: bool = False):
        keys = dict_str.split(split_str)

        data_mapper = copy.deepcopy(self.config['data_mapper'])
        data_id_ptr = self.config['data_id_ptr']

        # Check the key and insert the obj id
        if (self.__check_keys_and_insert(data_mapper, keys, data_id_ptr + 1, dict_str, is_force) is True):
            # Save the obj
            out = io.BytesIO()
            pickle.dump(obj, out)
            out.seek(0)
            data = sqlite3.Binary(out.read())

            self.db.insert("object", {
                "id": data_id_ptr + 1,
                "data": data,
                "info": info
            })

            # Update mapper and id_ptr
            self.config['data_mapper'] = data_mapper
            self.config['data_id_ptr'] += 1

            return True
        elif (is_force):
            # Update the obj
            out = io.BytesIO()
            pickle.dump(obj, out)
            out.seek(0)
            data = sqlite3.Binary(out.read())

            self.db.update("object", {
                "data": data,
                "info": info
            }, f"WHERE id={self.__get_data_id(data_mapper, keys)}")

            return True

    def load_obj(self, dict_str: str, split_str: str = ".") -> object:
        keys = dict_str.split(split_str)

        data_id = self.__get_data_id(self.config['data_mapper'], keys)

        if (data_id == -1):
            raise KeyError(f"The path {dict_str} is not exists.")

        result = self.db.query("object", f"WHERE id={data_id}")[0]
        assert len(result) == 1

        out = io.BytesIO()
        out.write(result[0][1])
        out.seek(0)
        return pickle.load(out)

    # def save_csv(self):
    #     pass

    # def load_csv(self):
    #     pass
