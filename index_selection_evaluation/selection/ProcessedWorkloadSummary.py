import os
import pickle
import numpy as np
import copy
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import time
from line_profiler import LineProfiler
from functools import wraps
import torch
import logging
import sys

import constants
from .workload import Workload,ProcessedWorkload,Plan
from .summary_cost_evaluation import CostEvaluation
from .index import Index
from .cost_evaluation import CostEvaluation as oringeCostEvaluation
from .utils import load_checkpoint,save2checkpoint

from ChangeFormer.model import ChangeFormer
from ChangeFormer.dataset import Encoding
from ChangeFormer.database_util import collator,Batch

class ProcessedWorkloadSummary:

    def __init__(self,workload,cost_evaluation,config=constants.config):
        self.workload=workload
        self.config=config
        self.cost_and_plan=workload.cost_and_plan
        self.benchmark_name=workload.benchmark_name
        self.cost_evaluation=cost_evaluation
        
        self.pWorkload=load_checkpoint(f'{constants.dir_path}/full_processed_workload')
        if torch.cuda.is_available():
            self.device='cuda'
        else:
            self.device='cpu'
        self._get_indexable_column_operator()
        
        self.qco_cost_o=np.zeros([len(workload.queries),len(self.indexable_columns_name)+1,len(constants.node_log_tpye)])
        self.qco_reduce_o=np.zeros([len(workload.queries),len(self.indexable_columns_name)+1,len(constants.node_log_tpye)])  
        
        self.pre_multi_index_len=self.config['pre_multi_index_len']
        self.index_used_num=0
        self.qco_index_dict={}

        self.pre_multi_operator_len=self.config['pre_multi_operator_len']
        self.operatro_used_num=0
        self.qco_operator_dict={}

        self.qco_cost_t=np.zeros([len(workload.queries),self.pre_multi_operator_len])
        self.qco_reduce_t=np.zeros([len(workload.queries),self.pre_multi_index_len,self.pre_multi_operator_len])
        self.qco_reduce_o2t=np.zeros([len(workload.queries),len(self.indexable_columns_name)+1,self.pre_multi_operator_len])
        
        self.operator2operatordict=np.zeros([len(constants.node_log_tpye),self.pre_multi_operator_len])

        self.qco_reduce_cost_t=np.zeros([len(workload.queries),self.pre_multi_index_len,self.pre_multi_operator_len])
        
        self.qco_selfmask_t=np.zeros([self.pre_multi_index_len,self.pre_multi_operator_len])
        
        self.qco_reduce_cost_o2t=np.zeros([len(workload.queries),len(self.indexable_columns_name)+1,self.pre_multi_operator_len])

        self.use_selectivity_num=0
        self.can_reduce_num=0
        self.can_reduce_True_num=0
        self.reduce_dict={}
        self.qco_cluster_t=np.zeros([len(workload.queries),self.pre_multi_operator_len])
        self.t2o_mask=np.zeros([self.pre_multi_index_len,len(self.indexable_columns_name)+1])
        
        self.o2t_cost_mask=np.zeros([len(self.indexable_columns_name)+1,self.pre_multi_operator_len])
        
        self.full_query_batch=load_checkpoint(f'{constants.dir_path}/full_query_batch')
        self.pre_feature_len=self.full_query_batch.features.shape[1]
        
        self.summary_with_opt_cache=[]
        self._generate_qco()
        self._load_model()

    def _get_indexable_column_operator(self):
        self.indexable_columns=self.pWorkload.indexable_columns
        self.indexable_columns_dict=self.pWorkload.indexable_columns_dict
        self.indexable_columns_name=self.pWorkload.indexable_columns_name
        self.indexable_table_columns_name=self.pWorkload.indexable_table_columns_name

    def _generate_qco(self):
        for query_id,pplan in enumerate(self.pWorkload.processed_plans):
            self._parse_plan(query_id,pplan)
        
        for c,p in self.qco_index_dict.items():
            self.t2o_mask[p,c[0]]=1
    
    def _parse_plan(self,query_id:int, pplan:Plan):
        for i,node in enumerate(pplan.dfs_nodes):
            columns=()
            if len(node.columns)!=0:
                columns=tuple(node.columns)
            else:
                columns=tuple(node.join_columns)
            operator_type=constants.phy2log_map[node.nodeType]
            cost=node.est_cost
            
            if len(columns)==0:
                self.qco_cost_o[query_id,-1,operator_type]=cost
            elif len(columns)==1:
                can_index,ratio=self.can_reduce(node)
                co=columns[0]
                o_cost=self.qco_cost_o[query_id,co,operator_type]
                o_ratio=self.qco_reduce_o[query_id,co,operator_type]
                s_cost=o_cost+cost
                if s_cost!=0:
                    self.qco_reduce_o[query_id,co,operator_type]=(cost*ratio+o_cost*o_ratio)/s_cost
                    self.qco_cost_o[query_id,co,operator_type]=s_cost

            elif len(columns)>=2:
                o_key=(columns,operator_type)
                if o_key in self.qco_operator_dict:
                    o_i=self.qco_operator_dict[o_key]
                else:
                    o_i=self.operatro_used_num
                    self.operator2operatordict[operator_type,o_i]=1
                    self.qco_operator_dict[o_key]=o_i
                    self.operatro_used_num+=1

                self.qco_cluster_t[query_id,o_i]=cost
                
                o_t_cost=self.qco_cost_t[query_id,o_i]
                self.qco_cost_t[query_id,o_i]+=cost
                l=len(columns)
                for i in range(l):
                    can_index,ratio=self.can_reduce(node,index_columns=(columns[i]))
                    o_ratio=self.qco_reduce_o2t[query_id,i,o_i]
                    self.qco_reduce_o2t[query_id,i,o_i]=(cost*ratio+o_t_cost*o_ratio)/(cost+o_t_cost)
                    self.qco_reduce_cost_o2t[query_id,i,o_i]=cost*ratio
                    self.o2t_cost_mask[columns[i],o_i]=1
                    for j in range(l):
                        if i==j:
                            continue
                        c_i_t=(columns[i],columns[j])
                        if c_i_t in self.qco_index_dict:
                            ct=self.qco_index_dict[c_i_t]
                        else:
                            ct=self.index_used_num
                            self.qco_index_dict[c_i_t]=ct
                            self.index_used_num+=1
                        
                        can_index,ratio=self.can_reduce(node,index_columns=c_i_t)
                        o_ratio=self.qco_reduce_t[query_id,ct,o_i]
                        self.qco_reduce_t[query_id,ct,o_i]=(cost*ratio+o_t_cost*o_ratio)/(cost+o_t_cost)
                        self.qco_reduce_cost_t[query_id,ct,o_i]+=cost*ratio
                        self.qco_selfmask_t[ct,o_i]=1
                        
    
    def can_reduce(self,node,index_columns=None):
        self.can_reduce_num+=1
        
        node_type_id=constants.phy2log_map[node.nodeType]
        can_idexed=True
        is_indexed=node.nodeType in constants.indexed_type
        if node_type_id==0 or node_type_id==6 or node_type_id==1 or node_type_id==2:

            if len(node.columns)>0:
                column=node.columns[0]
                if self.pWorkload.indexable_columns[column].column_type not in constants.can_index_column_tpye:
                    can_idexed=False
            for op in node.operatores:
                if op not in constants.can_index_operator_str:
                    can_idexed=False
        else:
            can_idexed=False
        if node.has_or and len(node.columns)>1:
            can_idexed=False
        if is_indexed:
            can_idexed=False
        
        if not can_idexed:
            return False,0
        else:
            if node_type_id==0 or node_type_id==6 or node_type_id==2:
                
                if index_columns is None:
                    origin_ratio=1-node.act_selective
                    ratio=origin_ratio*constants.log_type_r[node_type_id]+constants.log_type_b[node_type_id]
                    
                    ratio=max(ratio,0)
                    ratio=min(ratio,1)
                    return True,ratio
                else:
                    self.can_reduce_True_num+=1
                    if node_type_id in self.reduce_dict:
                        self.reduce_dict[node_type_id]+=1
                    else:
                        self.reduce_dict[node_type_id]=1
                    used_selectivity=node.act_selective
                    if index_columns in node.index2selectivity:
                        self.use_selectivity_num+=1
                        used_selectivity=node.index2selectivity[index_columns]
                    origin_ratio=1-used_selectivity
                    ratio=origin_ratio*constants.log_type_r[node_type_id]+constants.log_type_b[node_type_id]
                    
                    ratio=max(ratio,0)
                    ratio=min(ratio,1)
                    return True,ratio
            else:

                return True,0.9

    def _load_model(self):
        mpd={'lr': 0.0005789346253890466,
            'bs': 256,
            'epochs': 30,
            'clip_size': 8,
            'embed_size': 64,
            'pred_hid': 128,
            'ffn_dim': 256,
            'head_size': 16,
            'n_layers': 1,
            'join_schema_head_size': 2,
            'attention_dropout_rate': 0.06414118211287106,
            'join_schema_layers': 4,
            'dropout': 0.07326263251335534,
            'sch_decay': 0.22424663955914978,
            'device':'cuda:0'}
        self.model = ChangeFormer(emb_size=mpd['embed_size'], ffn_dim=mpd['ffn_dim'], \
                    head_size=mpd['head_size'], join_schema_head_size=mpd['join_schema_head_size'], \
                    dropout=mpd['dropout'],attention_dropout_rate=mpd['attention_dropout_rate'], \
                    n_layers=mpd['n_layers'],join_schema_layers=mpd['join_schema_layers'],pred_hid=mpd['pred_hid'])
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(f"./ChangeFormer/dataset/model.pth",map_location=self.device))
        self.model.eval()
        

    def _can_summary(self,template_l,indexes_feature):
        for i in range(40):
            self.full_query_batch.index_features[:,i]=indexes_feature[i]
        res=self.model(self.full_query_batch)
        _, pred_labels = torch.max(res.data, 1)
        pred_labels=pred_labels[template_l]
        self.full_query_batch.index_features[:,:]=self.none_column_idx
        template,simple=[],[]
        for i,f in enumerate(template_l):
            if pred_labels[i]:
                template.append(f)
            else:
                simple.append(f)
        return template,simple

    def _get_indexes(self,indexes):
        indexes_o=np.zeros([len(self.indexable_columns)+1])
        indexes_t=np.zeros([self.pre_multi_index_len])
        indexes_feature=torch.tensor([self.none_column_idx]*40,dtype=torch.float32)
        for i,idx in enumerate(indexes):
            if idx.is_single_column():
                if idx.columns[0].name in self.indexable_columns_dict.keys():
                    index_0=self.indexable_columns_dict[idx.columns[0].name]
                    indexes_o[index_0]=1
                    if i < 20:
                        indexes_feature[i*2]=index_0
            else:
                if idx.columns[0].name in self.indexable_columns_dict.keys() and idx.columns[1].name in self.indexable_columns_dict.keys():
                    index_0=self.indexable_columns_dict[idx.columns[0].name]
                    index_1=self.indexable_columns_dict[idx.columns[1].name]
                    if i < 20:
                        indexes_feature[i*2]=index_0
                        indexes_feature[i*2+1]=index_1
                    cid=(index_0,index_1)
                    if cid in self.qco_index_dict.keys():
                        indexes_t[self.qco_index_dict[cid]]=1
                    else:
                        if idx.columns[0].name in self.indexable_columns_dict.keys():
                            indexes_o[self.indexable_columns_dict[idx.columns[0].name]]=1
        return indexes_o,indexes_t,indexes_feature


    def _binary_search_loop(self, workload_list):
        low, high = 0, min((self.template_query_num-1),len(workload_list)-1)
        if workload_list[high]<=self.template_query_num-1:
            return workload_list[:(high+1)],workload_list[(high+1):]
        if workload_list[low]>self.template_query_num-1:
            return [],workload_list
        while low <= high:
            mid = int((low + high) / 2)
            if workload_list[mid] < self.template_query_num-1:
                low = mid + 1
            elif workload_list[mid] > self.template_query_num-1:
                high = mid - 1
            else:
                return workload_list[:(mid+1)],workload_list[(mid+1):]
        return workload_list[:(high+1)],workload_list[(high+1):]
    
    def get_cost_ml(self, workload_list, indexes):
        if len(workload_list)==0:
            return []
        indexes_o,indexes_t,indexes_feature=self._get_indexes(indexes)
        mask_t2o=np.sum(self.t2o_mask[indexes_t!=0,:],axis=0)
        indexes_o_mask=((indexes_o+mask_t2o)!=0)
        indexes_t_mask=((indexes_t)!=0)
        template_l,simple_l=self._binary_search_loop(workload_list)
        if len(template_l)==0:
            evul_list=[]
            q=simple_l
        elif len(template_l)<=2:
            evul_list=template_l
            q=simple_l
        else:
            evul_list,q=self._can_summary(template_l,indexes_feature)
            q.extend(simple_l)
        reduce_cost_o=self.qco_reduce_cost_o[q,:][:,indexes_o_mask]
        tem_cost_t=self.qco_reduce_cost_t[q,:,:][:,indexes_t_mask,:]
        tem_cost_o2t=self.qco_reduce_cost_o2t[q,:,:][:,indexes_o_mask,:]
        if tem_cost_t.size==0:
            reduce_cost_t=np.zeros((len(q),self.pre_multi_operator_len))
        else:
            reduce_cost_t=np.amax(self.qco_reduce_cost_t[q,:,:][:,indexes_t_mask,:],axis=-2)
        if tem_cost_o2t.size==0:
            reduce_cost_o2t=np.zeros((len(q),self.pre_multi_operator_len))
        else:
            reduce_cost_o2t=np.amax(self.qco_reduce_cost_o2t[q,:,:][:,indexes_o_mask,:],axis=-2)
        reduce_cost_t=np.maximum(reduce_cost_t,reduce_cost_o2t)
        s_cost_list=np.clip((self.cost_list[q]-np.sum(reduce_cost_o,axis=1)-np.sum(reduce_cost_t,axis=1)),0,None)*self.weight_list[q]
        wk=Workload([self.summary_queries[i] for i in evul_list])
        t_cost_list=self.cost_evaluation.calculate_cost(wk, indexes, store_size=True)
        return self._merge(evul_list,q,t_cost_list,s_cost_list,)
        
    
    def _merge(self,t_list,s_list,t_cost_list,s_cost_list):
        cost_list=[]
        t=0
        s=0
        while t<len(t_list) or s<len(s_list):
            if t<len(t_list) and s>=len(s_list):
                cost_list.extend(t_cost_list[t:])
                break
            elif t>=len(t_list) and s<len(s_list):
                cost_list.extend(s_cost_list[s:])
                break
            else:
                if t_list[t]<s_list[s]:
                    cost_list.append(t_cost_list[t])
                    t+=1
                else:
                    cost_list.append(s_cost_list[s])
                    s+=1
        assert len(cost_list)==len(t_cost_list)+len(s_cost_list)
        return cost_list

    def get_summary_workload(self):
        self._summary_workload()
        summary_queries=[]
        summary_list=[q[0] for q in self.templates]
        for query_list in self.templates:
            q=self.workload.queries[query_list[0]]
            s_cost=sum([self.cost_and_plan[i][0] for i in query_list])
            q.weight=s_cost/self.cost_and_plan[query_list[0]][0]
            summary_queries.append(q)
        summary_queries.extend([i for i in self.simple_query])
        summary_list.extend([i for i in self.simple_query_list])
        
        self.weight_list=np.array([q.weight for q in summary_queries])
        self.cost_list=np.array([self.cost_and_plan[i][0] for i in summary_list])
        self.total_cost=np.sum(self.cost_list)
        self.qco_reduce_cost_o=np.sum(self.qco_cost_o[summary_list,:,:]*self.qco_reduce_o[summary_list,:,:],axis=2)
        self.qco_reduce_cost_t=self.qco_reduce_cost_t[summary_list,:,:]
        self.qco_reduce_cost_o2t=self.qco_reduce_cost_o2t[summary_list,:,:]
        self.qco_reduce_o=self.qco_reduce_o[summary_list,:,:]
        self.qco_reduce_t=self.qco_reduce_t[summary_list,:,:]
        self.qco_reduce_o2t=self.qco_reduce_o2t[summary_list,:,:]
        self.qco_cost_o=self.qco_cost_o[summary_list,:,:]
        self.qco_cost_t=self.qco_cost_t[summary_list,:]
        self.add_node_cost=np.sum(self.qco_cost_o,axis=(1,2))+np.sum(self.qco_cost_t,axis=1)
        t_queries=summary_list[:self.template_query_num]
        self.full_query_batch.features=self.full_query_batch.features[t_queries,:,:]
        self.full_query_batch.attention_bias=self.full_query_batch.attention_bias[t_queries,:,:]
        self.full_query_batch.join_schema_bias=self.full_query_batch.join_schema_bias[t_queries,:,:]
        self.full_query_batch.heights=self.full_query_batch.heights[t_queries,:]
        self.none_column_idx=len(self.pWorkload.indexable_columns_dict)
        self.full_query_batch.index_features=self.full_query_batch.index_features[t_queries,:]
        self.full_query_batch.index_features[:,:]=self.none_column_idx
        self.full_query_batch.to(self.device)
        self.copy_full_query_batch=copy.deepcopy(self.full_query_batch)
        self.summary_workload=Workload(summary_queries)
        self.summary_queries=summary_queries
        return self.summary_workload

    def _summary_workload(self):
        
        f_pattern=re.compile(r'from')
        d_pattern=re.compile(r',')
        l=len(self.workload.queries)
        qco_flatten=self.qco_cost_o.reshape(l,-1)
        
        muf=self.qco_cluster_t
        qco_flatten=np.concatenate((qco_flatten,muf),axis=1)
        
        norms = np.linalg.norm(qco_flatten, axis=1, keepdims=True)
        
        qco_flatten = qco_flatten / norms
        self.threshold=self.config['threshold']
        

        complex_query=[]
        simple_query=[]
        
        templates=[]
        template_co=[]

        templates_simple=[]
        template_co_simple=[]
        for i,q in enumerate(self.workload.queries):
            query=q.text.lower()
            if len(f_pattern.findall(query))<=1:
                s=query.find('from')
                e=query.find('where')
                if len(d_pattern.findall(query[s:e]))<=1:
                    simple_query.append(i)
                    queryf=qco_flatten[i]
                    exist=False
                    for j,template in enumerate(template_co_simple):
                        
                        if np.sum(np.abs(queryf-template))<=self.threshold:
                            templates_simple[j].append(i)
                            exist=True
                            break
                    if not exist:
                        template_co_simple.append(queryf)
                        templates_simple.append([i])
                    continue
            
            complex_query.append(i)
            queryf=qco_flatten[i]
            exist=False
            for j,template in enumerate(template_co):
                
                if np.sum(np.abs(queryf-template))<=self.threshold:
                    templates[j].append(i)
                    exist=True
                    break
            if not exist:
                template_co.append(queryf)
                templates.append([i])
        self.templates=templates
        self.template_co=template_co
        self.complex_query=complex_query

        self.simple_query=[]
        self.simple_query_list=[]
        
        for query_list in templates_simple:
            q=self.workload.queries[query_list[0]]
            s_cost=sum([self.cost_and_plan[i][0] for i in query_list])
            q.weight=s_cost/self.cost_and_plan[query_list[0]][0]
            self.simple_query.append(q)
            self.simple_query_list.append(query_list[0])

        
        self.template_query_num=len(self.templates)
        self.summaried_query_num=len(self.templates)+len(self.simple_query)


class TestSummary:
    def __init__(self, db_connector):
        self.db_connector=db_connector
        self.db_connector.drop_indexes()
        
        self.cost_evaluation=CostEvaluation(self.db_connector)
class GetCostEvaluation:
    def __init__(self, db_connector):
        self.db_connector=db_connector
        self.db_connector.drop_indexes()
        self.cost_evaluation=oringeCostEvaluation(self.db_connector)


