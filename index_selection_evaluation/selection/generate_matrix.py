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
import json
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import torch
import logging
import sys

from selection import constants
from selection.benchmark import Benchmark
from selection.dbms.hana_dbms import HanaDatabaseConnector
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.query_generator import QueryGenerator
from selection.selection_algorithm import AllIndexesAlgorithm, NoIndexAlgorithm
from selection.table_generator import TableGenerator
from selection.workload import Workload,ProcessedWorkload
from selection.what_if_index_creation import WhatIfIndexCreation
from selection.summary_cost_evaluation import CostEvaluation
from selection.index import Index
from selection.cost_evaluation import CostEvaluation as oringeCostEvaluation
from selection.utils import save2checkpoint,load_checkpoint
from ChangeFormer.dataset import Encoding
from ChangeFormer.database_util import collator,Batch

def func_line_time(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        func_return = f(*args, **kwargs)
        lp = LineProfiler()
        lp_wrap = lp(f)
        lp_wrap(*args, **kwargs) 
        lp.print_stats() 
        return func_return 
    return decorator 


class CostPlanGenerate:
    def __init__(self, config, cost_estimation="actual_runtimes",is_load=True,re_compress=True) -> None:
        self.config=config
        self.cost_estimation=cost_estimation
        self.is_load=is_load
        self.re_compress=re_compress
        
        self.dir_path=self.config['dir_path']
        os.makedirs(self.dir_path,exist_ok=True)

        self.dbms_class=PostgresDatabaseConnector
        self.generating_connector = self.dbms_class(None, autocommit=True)
        self.table_generator = TableGenerator(
                    self.config["benchmark_name"], self.config["scale_factor"], self.generating_connector
                )
        self.database_name = self.table_generator.database_name()
        self.db_connector = PostgresDatabaseConnector(self.database_name)
        self.max_num=config['max compress num']

    def generate_query(self, times=1,random_select=False):
        if 'times' in self.config.keys() and times==1:
            times=self.config['times']
            self.times=times
        path=f'{self.dir_path}/full_query'
        if self.is_load and os.path.exists(path):
            with open(path,'rb') as f:
                self.workload=pickle.load(f)
            logging.info(f'load {len(self.workload.queries)} queries')
            self.cost_and_plan=self.workload.cost_and_plan
            self.total_cost=sum([cap[0] for cap in self.cost_and_plan])
        else:
            query_generator = QueryGenerator(
                        self.config["benchmark_name"],
                        self.config["scale_factor"],
                        self.db_connector,
                        self.config["queries"],
                        self.table_generator.columns,
                        self.config['random_queries'],
                        times=times
                    )
            if random_select:
                queries=query_generator.queries
                
                start_workload=copy.deepcopy(queries[0:len(self.config["queries"])])
                random_generated_queries=copy.deepcopy(queries[len(self.config["queries"])*times:])
                
                queries.sort(key=lambda q: q.nr)
                random_select_queries=[]
                l=[]
                for i in range(len(self.config["queries"])):
                    s=random.randint(0,times-1)
                    l.append(s+1)
                    random_select_queries.extend(queries[times*i:times*i+s])
                start_workload.extend(random_select_queries)
                start_workload.extend(random_generated_queries)
                self.workload = Workload(start_workload)
                logging.info(f'random select {len(start_workload)-len(random_generated_queries)} queries')
                logging.info(f'workload has {len(start_workload)} queries')
                logging.info(f'queries in each template: {l}')
            else:
                self.workload = Workload(query_generator.queries)
            logging.info(f'workload has {len(self.workload.queries)} queries')
            self.generate_cost_and_plan()
            with open(path,'wb') as f:
                pickle.dump(self.workload,f)
        
    def show_workload(self):
        logging.debug(f'workload.queries has {len(self.workload.queries)} queries')
        for i in range(self.times):
            logging.debug(self.workload.queries[0+i*len(self.config['queries'])].text)
    def generate_cost_and_plan(self,time_out=1000*60*15):
        self.db_connector.drop_indexes()
        self.db_connector.create_statistics()
        self.db_connector.commit()
        
        cost_and_plan=[]
        self.total_cost=0

        for i,query in enumerate(self.workload.queries):
            if self.cost_estimation=="whatif":
                plan=self.db_connector._get_plan(query)
                result=(plan["Total Cost"], plan)
                self.total_cost+=plan["Total Cost"]
            elif self.cost_estimation=="actual_runtimes":
                result=self.db_connector.exec_query(query,timeout=time_out)
                if type(result) == tuple:

                    self.total_cost+=result[0]
                    logging.info(f'query {i+1} has complete, spend {result[0]/1000} s')
                else:
                    logging.info(f'query {i+1} execute time is too long, fill with empty tuple')
                    result=(0,0)   
            cost_and_plan.append(result)
        
        self.workload.cost_and_plan=cost_and_plan
        self.workload.benchmark_name=self.database_name

        self.cost_and_plan=cost_and_plan
        self._generate_actual_plan()

    def _generate_actual_plan(self):
        plans=[]
        for query in self.workload.queries:
            plan=self.db_connector.exec_query(query)
            plans.append(plan)
        pWorkload=ProcessedWorkload(constants.config)
        pWorkload.add_from_triple(self.workload.queries,[],plans)
        save2checkpoint(f'{constants.dir_path}/full_processed_workload',pWorkload)
        ecd=Encoding(pWorkload)
        full_query_features=[]
        for plan in pWorkload.processed_plans:
            fet=ecd.get_plan_encoder(plan,set())
            full_query_features.append(fet)
        batch,_=collator((full_query_features,0))
        save2checkpoint(f'{constants.dir_path}/full_query_batch',batch) 

    def _get_cost_and_plan(self,workload):
        
        self.db_connector.drop_indexes()
        self.db_connector.create_statistics()
        self.db_connector.commit()
        
        cost_and_plan=[]
        
        for i,query in enumerate(workload.queries):
            if self.cost_estimation=="whatif":
                plan=self.db_connector._get_plan(query)
                result=(plan["Total Cost"], plan)
            elif self.cost_estimation=="actual_runtimes":
                raise NotImplementedError
            cost_and_plan.append(result)
        return cost_and_plan

    def compress(self):
        self.compressed_workload_dict={}
        if not self.re_compress:
            logging.info('Warning: will not re compress')
            for algorithm in self.config['compress_algorithm']:
                cpath=f'{self.dir_path}/compressed_by_{algorithm}'
                with open(cpath,'rb') as f:
                    self.compressed_workload_dict[algorithm]=pickle.load(f)
        for algorithm in self.config['compress_algorithm']:
            logging.info(algorithm)
            start=time.time()
            self._compress_once(algorithm)
            logging.info(f'{algorithm} complete time is {time.time()-start}s')
            logging.info('\n')

    
    def _compress_once(self,algorithm):
        if algorithm=='sample':
            alg=SampleCompress(self.config)
            compressed_workload=alg.compress(self.workload,self.cost_and_plan,self.max_num)
            self.sample_picked_queries=alg.picked_queries
            self.reCost(compressed_workload)
        elif algorithm=='nothing':
            compressed_workload=self.workload

        else:
            raise NotImplementedError
        
        self.compressed_workload_dict[algorithm]=compressed_workload
        cpath=f'{self.dir_path}/compressed_by_{algorithm}'
        with open(cpath,'wb') as f:
            pickle.dump(compressed_workload,f)
    
    def reCost(self,compressed_workload):
        if self.cost_estimation=="actual_runtimes":
            raise NotImplementedError
        compress_cost=0
        for query in compressed_workload.queries:
            compress_cost+=self.db_connector._get_plan(query)["Total Cost"]
        ratio=self.total_cost/compress_cost
        for query in compressed_workload.queries:
            query.weight=query.weight*ratio
    def close(self):
        self.db_connector.commit()
        self.db_connector.close()


class SampleCompress:
    def __init__(self,config):
        self.config=config
        logging.debug('SampleCompress init')
    
    
    def pick_query(self):
        picked_queries=[]
        l=len(self.workload.queries)-1
        for _ in range(self.max_num):
            picked_queries.append(random.randint(0,l))
        self.picked_queries=picked_queries
        return picked_queries
    
    def compress(self, workload, cost_and_plan,max_num):
        self.workload=workload
        self.max_num=max_num
        picked_query=self.pick_query()
        weight=len(workload.queries)/len(picked_query)
        queries=[]
        for i in range(len(picked_query)):
            query=copy.deepcopy(workload.queries[picked_query[i]])
            query.weight=weight
            queries.append(query)
        compressed_workload=Workload(queries)
        logging.debug(f'Sample compress complete, workload {compressed_workload.queries}')
        return compressed_workload



if __name__=='__main__':
    cap=CostPlanGenerate(constants.config,is_load=False)
    cap.generate_query()
    cap.compress()
    cap.close()
