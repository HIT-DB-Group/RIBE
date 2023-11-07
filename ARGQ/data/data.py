from train_utils import Postgre
from settings import Settings
import pickle
import psycopg2
import math
import random
import sys
from collections import OrderedDict
import re
import os
from typing import Dict, NamedTuple, Optional, Tuple, Any, List
import sys
sys.path.append("..")


class DataProcess:
    def __init__(self):
        self.load()

    def load(self):
        if Settings.predicate_path.exists():
            with Settings.predicate_path.open("rb") as f:
                self.predicate = pickle.load(f)
            if self.predicate.data_name == Settings.data_name:
                return
        self.predicate = Predicate(data_name=Settings.data_name)
        self.get_predicate()

    def dump(self):
        with Settings.predicate_path.open("wb") as f:
            pickle.dump(self.predicate, f)

    def get_predicate(self):
        database = Postgre(Settings.database_url)
        selectbase = "SELECT * FROM "
        with Settings.origin_sql_path.open("r") as f:
            for sql in f:
                for pred in re.finditer(Settings.regular_predicate, sql):
                    table_short = pred.group(1)
                    attribute = pred.group(2)
                    #operator = pred.group(3)
                    value = pred.group(4)
                    if attribute not in self.predicate.predicate.keys():
                        self.predicate.predicate[attribute] = dict()
                    self.predicate.predicate[attribute][value] = dict()
                    for operator in Settings.operators:
                        table = Settings.tables[Settings.tables_short.index(
                            table_short)]
                        sql = selectbase + table + " WHERE " + \
                            f"{attribute}{operator}{value};"
                        print(sql)
                        cardinality, _ = database.parser(sql)
                        self.predicate.predicate[attribute][value][operator] = cardinality
        self.dump()

class Predicate:
    def __init__(self, data_name):
        self.data_name = data_name
        # attribute:value::operator:cardinality
        self.predicate: Dict[str, Dict[str, Dict[str, int]]] = dict()


class Hash:
    def __init__(self, predicate):
        self.predicate = predicate
        attributes = self.predicate.keys()
        self.buckets = dict.fromkeys(attributes, None)
        self.col_info = dict.fromkeys(attributes, None)
        self.hash_num = dict.fromkeys(attributes, None)
        for attribute in attributes:
            values = self.predicate[attribute].keys()
            minval = int(min(values[col]))
            maxval = int(max(values[col]))
                hash_num = min(Constants.HASH_NUM, maxval-minval+1)
                self.col_info[col] = (minval, maxval)
                self.hash_num[col] = hash_num
        for col in Constants.colset:
            vals = values[col]
            if vals is not None:
                for val in vals:
                    self.add_(col, val)

    def get_key(self, col, value):
        (min, max) = self.col_info[col]
        key = math.ceil(self.hash_num[col]*(value-min+1)/(max-min+1)-1)
        return key

    def get_value(self, cards, col, key, op, card):
        key = key % self.hash_num[col]
        while key not in self.hash_bucket[col].keys():
            key = key+1
            key = key % self.hash_num[col]
        vals = self.hash_bucket[col][key]
        for v in vals:
            if cards[col][v][op] == int(card):
                return v
        value = random.choice(vals)
        return value

    def add_item(self, col, value):
        key = self.key_(col, value)
        if self.hash_bucket[col] == None:
            self.hash_bucket[col] = {}
        if key not in self.hash_bucket[col].keys():
            self.hash_bucket[col][key] = []
        self.hash_bucket[col][key].append(value) '''
