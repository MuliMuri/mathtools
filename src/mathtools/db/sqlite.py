#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   sqlite.py
@Time    :   2024/07/18 05:31:23
@Author  :   MuliMuri
@Version :   1.0
@Desc    :   SQLite database secondary package
'''


import sqlite3
import threading

from typing import Dict, List, Tuple


def singleton(cls):
    _instance = {}

    def inner(database):
        if (cls, database) not in _instance:
            _instance[(cls, database)] = cls(database)

        return _instance[(cls, database)]
    return inner


SQL_DICT = {
    "CreateTable": "CREATE TABLE IF NOT EXISTS '{table_name}' ({columns})",
    "CheckTableExists": "SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'",
    "InsertData": "INSERT INTO '{table_name}' ({columns}) VALUES ({values})",
    "SelectData": "SELECT * FROM '{table_name}' {condition}",
    "UpdateData": "UPDATE '{table_name}' SET {sets} {condition}",
    "DeleteData": "DELETE FROM '{table_name}' {condition}",
    "DropTable": "DROP TABLE IF EXISTS '{table_name}'"
}


@singleton
class SQLite():
    def __init__(self, database: str) -> None:
        """SQLite class

        Args:
            database (str): The name of database
        """
        self.database = database
        self.db = sqlite3.connect(database, check_same_thread=False)
        self.cursor = self.db.cursor()

        self.lock = threading.Lock()

    def __format_str(self, datas: List):
        for i, data in enumerate(datas):
            if (data is str):
                if "'" in data:
                    datas[i] = data.replace("'", "''")

        return ','.join("'{0}'".format(x) for x in datas)

    def create_table(self, table_name: str, columns: List):
        """Create a new table

        Args:
            table_name (str): The name of table
            columns (list): list of columns name
        """
        sql = SQL_DICT['CreateTable']
        columns_str = self.__format_str(columns)

        sql = sql.format(
            table_name=table_name,
            columns=columns_str)

        self.execute(sql)

    def is_table_exists(self, table_name: str) -> bool:
        """Check table exists

        Args:
            table_name (str): The table name

        Returns:
            bool: If exists then return True
        """
        sql = SQL_DICT['CheckTableExists']
        sql = sql.format(table_name=table_name)

        result = self.execute(sql)

        if (len(result) != 0):
            return True

        return False

    def insert(self, table_name: str, data: Dict):
        """Insert a data to table

        Args:
            table_name (str): The name of you want to inserted table's name
            data (dict): data dictionary, key is column's name, value is data
        """
        sql = SQL_DICT['InsertData']
        columns_str = self.__format_str(list(data.keys()))
        values = tuple(data.values())
        holder_str = ",".join(["?" for i in range(len(data))])

        sql = sql.format(table_name=table_name,
                         columns=columns_str,
                         values=holder_str
                         )

        self.execute(sql, values)

    def query(self, table_name: str, condition: str = "", is_contain_column_name: bool = False) -> tuple:
        """Query data

        Args:
            table_name (str): The name of table
            condition (str, optional): Conditions of query. Defaults to "". If you want to add condition, \
                please add it like 'WHERE item1 = 2', must add 'WHERE'
            is_contain_column_name (bool): If its true, return (results, columns)

        Returns:
            tuple: results of query
        """
        sql = SQL_DICT['SelectData']
        sql = sql.format(table_name=table_name,
                         condition=condition
                         )

        ret = self.execute(sql, is_column=True)
        if (ret is not None):
            results, columns = ret

        if (not is_contain_column_name):
            return (results, )

        return (results, columns)

    def update(self, table_name: str, column_value: Dict, condition: str = ""):
        """Update a item

        Args:
            table_name (str): The name of table
            column_value (dict): key:column_name, value:new_value example:{'column1':'value1'}
            condition (str, optional): Conditions of update. Defaults to "". If you want to add condition, \
                please add it like 'WHERE item1 = 2', must add 'WHERE'.
        """
        sql = SQL_DICT['UpdateData']

        combined_list = [f"{k}={v}" for k, v in zip(list(column_value.keys()), ["?" for i in range(len(column_value))])]

        values = tuple(column_value.values())

        sql = sql.format(table_name=table_name,
                         sets=",".join(combined_list),
                         condition=condition
                         )

        self.execute(sql, values)

    def delete(self, table_name: str, condition: str = ""):
        """Delete a item

        Args:
            table_name (str): The name of table
            condition (str, optional): Conditions of delete. Defaults to "". If you want to add condition, \
                please add it like 'WHERE item1 = 2', must add 'WHERE'.
        """
        sql = SQL_DICT['DeleteData']
        sql = sql.format(table_name=table_name,
                         condition=condition
                         )

        self.execute(sql)

    def drop(self, table_name: str):
        """Delete a table

        Args:
            table_name (str): The name of table
        """
        sql = SQL_DICT['DropTable']
        sql = sql.format(table_name=table_name)

        self.execute(sql)

    def execute(self, sql: str, data_tuple: Tuple = (), is_column=False) -> List:
        """Execute custom sql

        Args:
            sql (str): sql
            is_column (bool, optional): Does the returned query result include column names. Defaults to False.

        Returns:
            List: Returned query results
        """
        with self.lock:
            try:
                self.cursor.execute(sql, data_tuple)

                self.db.commit()

                if (is_column):
                    column_name, _, _, _, _, _, _ = zip(*self.cursor.description)
                    return [self.cursor.fetchall(), column_name]

                return self.cursor.fetchall()
            except Exception as e:
                print(e)
                self.db.rollback()

        return None

    def __del__(self):
        self.db.close()
