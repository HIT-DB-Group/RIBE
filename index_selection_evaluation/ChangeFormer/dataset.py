import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import json
import pandas as pd
import sys, os, copy,logging
from collections import deque
import time
from scipy.stats import pearsonr
import pickle
import random


from ChangeFormer.database_util import *
from selection import constants
from selection.utils import *
from selection.workload import ProcessedWorkload


phy2log_map=constants.phy2log_map
indexed_type=constants.indexed_type
can_index_column_tpye=constants.can_index_column_tpye
can_index_operator_str=constants.can_index_operator_str

class Encoding:
    def __init__(self, pWorkload:ProcessedWorkload):
        self.pWorkload=pWorkload
        self.node_types2idx={'Unique':0,'Hash Join':1,'Bitmap Heap Scan':2,'Materialize':3,'SetOp':4,'Subquery Scan':5,'Aggregate':6,'BitmapAnd':7,'Gather Merge':8,'WindowAgg':9,'Sort':10,'Gather':11,'Index Scan':12,'Merge Join':13,'Bitmap Index Scan':14,'Nested Loop':15,'Index Only Scan':16,'CTE Scan':17,'Hash':18,'BitmapOr':19,'Limit':20,'Result':21,'Merge Append':22,'Append':23,'Group':24,'Seq Scan':25}

        tablesid2sort=[(degree+i*0.0001,i) for i,degree in enumerate(np.sum(self.pWorkload.join_schema,axis=0))]
        list.sort(tablesid2sort,key=lambda x:x[0],reverse=True)
        self.tablesid2idx={id:i for i,(_,id) in enumerate(tablesid2sort)}
        tablesExc=[ id for _,id in tablesid2sort]
        self.join_schema_bias=self.pWorkload.join_schema[tablesExc,:]
        self.join_schema_bias=self.join_schema_bias[:,tablesExc]

        self.tablesid2idx[None]=len(tablesid2sort)
        self.join_schema_bias=np.pad(self.join_schema_bias,((0,1),(0,1)),'constant',constant_values=0)

        self.none_column_idx=len(self.pWorkload.indexable_columns_dict)

        self.index_change2idx={(False,False):0,(True,False):1,(True,True):2,(False,True):3}

    def get_plan_encoder(self,plan,new_indexes):
        new_indexes=list(new_indexes)
        new_indexes_idx=[]
        for indexes in new_indexes:
            names=indexes._column_names()
            new_indexes_idx.append([self.pWorkload.indexable_columns_dict[name] for name in names])
        
        indexe_features=np.full(20*2,self.none_column_idx,dtype=float)
        for i,indexes in enumerate(new_indexes_idx):
            indexe_features[i*2]=indexes[0]
            if len(indexes)>1:
                indexe_features[i*2+1]=indexes[1]
        
        features=np.zeros((len(plan.dfs_nodes),12),dtype=float)
        for i,node in enumerate(plan.dfs_nodes):
            features[i][0]=self.node_types2idx[node.nodeType]
            features[i][1]=self.tablesid2idx[node.table]
            l=len(node.columns)
            for j in range(2):
                if j>=l:
                    features[i][2+j]=self.none_column_idx
                else:
                    features[i][2+j]=node.columns[j]
            
            l=len(node.join_tables)
            for j in range(3):
                if j>=l:
                    features[i][4+j]=self.tablesid2idx[None]
                elif node.join_tables[j]=='middle_tabble' or node.join_tables[j]=='notfound':
                    features[i][4+j]=self.tablesid2idx[None]
                else:
                    features[i][4+j]=self.tablesid2idx[node.join_tables[j]]
            l=len(node.join_columns)
            for j in range(3):
                if j>=l:
                    features[i][7+j]=self.none_column_idx
                else:
                    features[i][7+j]=node.join_columns[j]

            if plan.is_act:
                features[i][10]=node.act_cost
                features[i][11]=node.act_rows
            else:
                features[i][10]=node.est_cost
                features[i][11]=node.est_rows

        costmax=np.max(features[:,10])
        costmin=np.min(features[:,10])
        cardmax=np.max(features[:,11])
        cardmin=np.min(features[:,11])
        costabs=(costmax-costmin)
        cardabs=(cardmax-cardmin)
        if costabs==0:
            features[:,10]=1
        else:
            features[:,10]=(features[:,10]-costmin)/costabs
        
        if cardabs==0:
            features[:,11]=1
        else:
            features[:,11]=(features[:,11]-cardmin)/cardabs
        
        return {'features':features,'attention_bias':plan.adj,'heights':plan.heights,'indexe_features':indexe_features,'join_schema_bias':self.join_schema_bias,'length':len(plan.dfs_nodes)}

        
    def column_in_idxs(self,columns,indexes_idx):
        columns_set=set(columns)
        for idx in indexes_idx:
            if columns_set<=set(idx):
                return idx
        return None

    def index_change_column(self,node,new_indexes_idx):
        node_type_id=phy2log_map[node.nodeType]
        can_idexed=True
        index_idxs=None
        if node_type_id==0 or node_type_id==6 or node_type_id==1:
            for column in node.columns:
                if self.pWorkload.indexable_columns[column].column_type not in can_index_column_tpye:
                    can_idexed=False
            for op in node.operatores:
                if op not in can_index_operator_str:
                    can_idexed=False
            index_idxs=self.column_in_idxs(node.columns,new_indexes_idx)
            if index_idxs is None:
                can_idexed=False
        else:
            can_idexed=False
        
        if node.has_or and len(node.columns)>1:
            can_idexed=False
        if can_idexed:
            return index_idxs
        else:
            return None



class PlanChangeDataset(Dataset):
    def __init__(self, act_cache:dict):
        self.act_cache=act_cache
        self.queries_2_index_plan_dict={}
        self.queries_2_noindex_plan_dict={}
        self.queries_dataset={}
        self.collated_dicts=[]
        
        self.queries=[]
        self.indexes=[]
        self.plans=[]
        for k,v in act_cache.items():
            self.queries.append(k[0])
            self.indexes.append(k[1])
            self.plans.append(v)
        self.pWorkload=ProcessedWorkload(constants.config)
        self.pWorkload.add_from_triple(self.queries,self.indexes,self.plans)
        for query,index,pPlan in zip(self.queries,self.indexes,self.pWorkload.processed_plans):
            if query in self.queries_2_index_plan_dict:
                self.queries_2_index_plan_dict[query].append((index,pPlan))
            else:
                self.queries_2_index_plan_dict[query]=[(index,pPlan)]
            if len(index)==0:
                self.queries_2_noindex_plan_dict[query]=(index,pPlan)
        self.encoding=Encoding(self.pWorkload)
    
    def generate_data(self):
        datas_num=0
        change_num=0
        for query,ips in self.queries_2_index_plan_dict.items():
            all=0
            change=0
            datas=[]
            _,no_plan=self.queries_2_noindex_plan_dict[query]
            for ip in ips:
                all+=1
                if no_plan.join_order==ip[1].join_order:
                    label=0
                else:
                    change+=1
                    label=1
                collated=self.encoding.get_plan_encoder(no_plan,ip[0])
                datas.append((collated,label))
                self.collated_dicts.append((collated,label))
            self.queries_dataset[query]=datas
            datas_num+=all
            change_num+=change    

    def shuffle(self):
        random.shuffle(self.collated_dicts)
    def __len__(self):
        return len(self.collated_dicts)
    
    def __getitem__(self, idx):
        return self.collated_dicts[idx][0], self.collated_dicts[idx][1]
    
    def dump_collated_dicts(self,path):
        save2checkpoint(obj=self.collated_dicts,path=path)



class MiddleDataset(Dataset):
    def __init__(self, collated_dicts:list):
        self.collated_dicts=collated_dicts
    
    def __len__(self):
        return len(self.collated_dicts)
    
    def __getitem__(self, idx):
        return self.collated_dicts[idx][0], self.collated_dicts[idx][1]

def get_ts_vs(path,train_ratio):
    collated_dicts=load_checkpoint(path)
    train_len=int(train_ratio*len(collated_dicts))
    return MiddleDataset(collated_dicts[:train_len]),MiddleDataset(collated_dicts[train_len:])







