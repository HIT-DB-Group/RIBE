from .index import Index
import json
import numpy as np
import re
from collections import deque
import logging
from selection import constants
from cost_estimation.sample_estimator import SampleEstimator


class Workload:
    def __init__(self, queries):
        self.queries = queries
        
        
        self.config_path='./example_configs/config_common.json'
        try:
            with open(self.config_path) as f:
                index_config=json.load(f)
                self.exist_index=index_config["exist_index_table.column"]
        except FileNotFoundError:
            print(f'there is no exist index')
            self.exist_index=[]
        self.cost_and_plan=None
        self.benchmark_name=None

    def indexable_columns(self):
        indexable_columns = set()
        for query in self.queries:
            indexable_columns |= set(query.columns)

        return sorted(list(indexable_columns))

    def potential_indexes(self):
        indexable_columns=self.indexable_columns()
        indexable_no_exist_columns=[]
        for column in indexable_columns:
            if f'{column.table.name}.{column.name}' not in self.exist_index:
                indexable_no_exist_columns.append(column)
        return sorted([Index([c]) for c in indexable_no_exist_columns])


class Column:
    def __init__(self, name, column_type=None):
        self.name = name.lower()
        self.table = None
        self.column_type=column_type

    def __lt__(self, other):
        return self.name < other.name

    def __repr__(self):
        
        return f"C {self.table}.{self.name}"

    
    
    def __eq__(self, other):
        if not isinstance(other, Column):
            return False

        assert (
            self.table is not None and other.table is not None
        ), "Table objects should not be None for Column.__eq__()"

        return self.table.name == other.table.name and self.name == other.name

    def __hash__(self):
        return hash((self.name, self.table.name))


class Table:
    def __init__(self, name):
        self.name = name.lower()
        self.columns = []

    def add_column(self, column):
        column.table = self
        self.columns.append(column)

    def add_columns(self, columns):
        for column in columns:
            self.add_column(column)

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, Table):
            return False

        return self.name == other.name and tuple(self.columns) == tuple(other.columns)

    def __hash__(self):
        return hash((self.name, tuple(self.columns)))

class Dataset:
    def __init__(self, name):
        self.name = name
        self.tables = []

class Query:
    def __init__(self, query_id, query_text, columns=None):
        self.nr = query_id
        self.text = query_text
        self.weight=1.0

        
        if columns is None:
            self.columns = []
        else:
            self.columns = columns

    def __repr__(self):
        return f"Q{self.nr} weight:{self.weight}"
    
    def set_weight(self,weight):
        self.weight=weight

node_phy_tpye=constants.node_phy_tpye
node_log_tpye=constants.node_log_tpye
key_list=constants.key_list
join_key_list=constants.join_key_list
filter_list=constants.filter_list
phy2log_map=constants.phy2log_map

column_tpye=constants.column_tpye
operator_str=constants.operator_str
can_index_column_tpye=constants.can_index_column_tpye

max_node_len=constants.max_node_len

class ProcessedWorkload:
    def __init__(self,config=None) :
        self.config=config
        
        self.queries=[]
        self.indxes=[]
        self.plans=[]
        self.init_processed_data()
        self.statistics=None
        if config is not None:
            self.load_statistics(config['stastics_path'],f'indexselection_{config["benchmark_name"]}___{config["scale_factor"]}')
            self.estimator=SampleEstimator(f'indexselection_{config["benchmark_name"]}___{config["scale_factor"]}')
        
    def load_statistics(self,path,dbname):
        with open(path,'r') as f:
            statistics=json.load(f)
        self.statistics=statistics[dbname]
   
    def init_processed_data(self):
        self.tables=[]
        self.indexable_columns=[]
        self.processed_plans=[]
    
    def init_columns_tables(self):
        indexable_columns = set()
        for query in self.queries:
            indexable_columns |= set(query.columns)
        self.indexable_columns=sorted(list(indexable_columns))
        
        tables = set()
        
        self.column2tablename={}
        for i,column in enumerate(self.indexable_columns):
            tables.add(column.table.name)
            self.column2tablename[i]=column.table.name

        self.tables=sorted(list(tables))
        self.column2tableid={}
        for cid,tname in self.column2tablename.items():
            self.column2tableid[cid]=self.tables.index(tname)

        self.indexable_columns_dict={col.name:i for i,col in enumerate(self.indexable_columns)}
        self.indexable_columns_name=[i.name for i in self.indexable_columns]
        self.indexable_table_columns_name=[f'{i.table}.{i.name}' for i in self.indexable_columns]
        

        
    def add_from_triple(self, queries:list, indexes:list,plans:list):
        
        self.init_processed_data()
        self.queries.extend(queries)
        self.indxes.extend(indexes)
        self.plans.extend(plans)
        self.init_columns_tables()
        
        self.join_schema=np.zeros([len(self.tables),len(self.tables)])
        
        for query,index,plan in zip(queries,indexes,plans):
            self.heights=np.zeros(max_node_len)
            self.adj=np.zeros([max_node_len,max_node_len])
            self.dfs_nodes=[]
            self.join_order=[]
            self.is_act='Actual Rows' in plan
            treeNode,_,_=self._parse_plan(plan,height=1,path=[])
            pl=len(self.dfs_nodes)
            self.heights=self.heights[:pl]
            self.adj=self.adj[:pl,:pl]
            self.processed_plans.append(Plan(query,index,treeNode,self.heights,self.adj,self.dfs_nodes,self.join_order,plan,self.is_act))
            
        
    def _parse_plan(self,plan:dict,height:int,path:list):
        # Add your code to parse the plan, there is an example in "index_selection_evaluation/workload_compress/workload/example_plan.pkl"
        pass
    

class Plan:
    def __init__(self,query,indexes,rootNode,heights,adj,dfs_nodes,join_order,js_plans,is_act):
        self.query=query
        self.indexes=indexes
        self.rootNode=rootNode
        
        self.heights=heights
        self.adj=adj
        self.dfs_nodes=dfs_nodes
        self.join_order=join_order
        self.js_plans=js_plans
        self.is_act=is_act 
        
class TreeNode:
    def __init__(self, nodeType,est_rows):
        self.nodeType = nodeType
        self.logType=None
        
        self.numFilterDict = []
        self.strFilterDict = []
        self.columnFilterDict = []
        self.has_or=False

        self.logicPredicate=[]
        
        self.index2selectivity={}

        self.table = None
        self.join_tables=[]
        
        self.join_columns = []
        self.columns=[]
        self.operatores=[]

        self.est_rows=est_rows
        self.act_rows=None
        
        self.est_cost = None
        self.act_cost = None

        self.est_selective=None
        self.act_selective=None
        
        self.children = []
        self.parent = None
        self.feature = None
        self.use_index=False
        
    def addChild(self,treeNode):
        self.children.append(treeNode)
    
    def __str__(self):
        res=f'{self.nodeType} with plan rows {self.est_rows}, columns {self.columns}'
        if self.table is not None:
            res+=f', self.table {self.table}'
        if self.est_cost is not None:
            res+=f', self.est_cost {self.est_cost:.1e}'
        if self.act_cost is not None:
            res+=f', self.act_cost {self.act_cost:.1e}'

        if len(self.join_tables)>0:
            res+=f', self.join_tables {self.join_tables}'
        if len(self.join_columns) >0:
            res+=f', self.join_columns {self.join_columns}'
        if len(self.numFilterDict) >0:
            res+=f', numFilterDict {self.numFilterDict}'
        if len(self.strFilterDict) >0:
            res+=f', strFilterDict {self.strFilterDict}'
        if len(self.columnFilterDict) >0:
            res+=f', columnFilterDict {self.columnFilterDict}'
        if self.est_selective is not None:
            res+=f', self.est_selective {self.est_selective:.1e}'
        if self.act_selective is not None:
            res+=f', self.act_selective {self.act_selective:.1e}'
        
        return res

    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def print_nested(node, indent = 0): 
        print('--'*indent+ '{} with plan rows{}, has {} children'.format(node.nodeType, node.est_rows, len(node.children)))
        for k in node.children: 
            TreeNode.print_nested(k, indent+1)







