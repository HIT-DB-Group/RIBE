import logging
from time import time

from .what_if_index_creation import WhatIfIndexCreation
import pickle


class CostEvaluation:
    def __init__(self, db_connector, cost_estimation="whatif",full_workload_path=None,log_plan=False):
        logging.debug("Init cost evaluation")
        self.db_connector = db_connector
        self.cost_estimation = cost_estimation
        logging.info("Cost estimation with " + self.cost_estimation)
        self.what_if = WhatIfIndexCreation(db_connector)
        self.current_indexes = set()
        self.cost_requests = 0
        self.cache_hits = 0

        self.relevant_indexes_cache = {}

        self.cache = {}
        self.s_cache={}
        self.completed = False
        
        
        self.log_plan=log_plan
        self.query_indexes_plan=[]
        self.full_workload_path=full_workload_path
        self.index_cost_list=[]
        self.alg_name=None
        self.compress_name=None
        self.have_run_calculate_cost_long=False 
        if full_workload_path is not None:
            with open(full_workload_path,'rb') as f:
                self.full_workload=pickle.load(f)

    def set_alg_name(self,name):
        self.alg_name=name
    def set_compress_name(self,name):
        self.compress_name=name
    
    

    def estimate_size(self, index):        
        result = None
        for i in self.current_indexes:
            if index == i:
                result = i
                break
        if result:
            
            if not index.estimated_size:
                index.estimated_size = self.what_if.estimate_index_size(result.hypopg_oid)
        else:
            self._simulate_or_create_index(index, store_size=True)

    def which_indexes_utilized_and_cost(self, query, indexes):
        self._prepare_cost_calculation(indexes, store_size=True)

        plan = self.db_connector.get_plan(query)
        cost = plan["Total Cost"]
        plan_str = str(plan)

        recommended_indexes = set()

        
        for index in self.current_indexes:
            assert (
                index in indexes
            ), "Something went wrong with _prepare_cost_calculation."

            if index.hypopg_name not in plan_str:
                continue
            recommended_indexes.add(index)

        return recommended_indexes, cost

    
    def analyze_est_and_act(self,workload,indexes):
        assert (
            self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        cost_estimation=self.cost_estimation
        self.cost_estimation="whatif"
        
        
        whatif_plans=[]
        whatif_costs=[]
        self._prepare_cost_calculation(indexes)
        for query in workload.queries:
            whatif_plan=self.db_connector._get_plan(query)
            whatif_plans.append(whatif_plan)
            whatif_cost=whatif_plan["Total Cost"]*query.weight
            whatif_costs.append(whatif_cost)
        self._prepare_cost_calculation([])
        
        
        self.cost_estimation="actual_runtimes"
        
        act_plans=[]
        act_costs=[]
        self._prepare_cost_calculation(indexes)
        for query in workload.queries:
            act_cost,act_plan=self.db_connector.exec_query(query)
            act_plans.append(act_plan)
            act_costs.append(act_cost)
        self._prepare_cost_calculation([])

        self.cost_estimation=cost_estimation
        return whatif_costs,whatif_plans,act_costs,act_plans

    
    def calculate_cost(self, workload, indexes, store_size=False):
        assert (
            self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        
        self._prepare_cost_calculation(indexes, store_size=store_size)
        total_cost = 0
        
        
        for query in workload.queries:
            self.cost_requests += 1
            total_cost += self._request_cache(query, indexes)
        
        return total_cost


    def dump_plan(self,benchmark_name):
        if self.alg_name is not None:
            filePath=f'./workload_compress/plan_change/{benchmark_name}_{self.alg_name}_query_indexes_plan.pkl'
            with open(filePath,'wb') as f:
                pickle.dump(self.query_indexes_plan,f)
        else:
            print('the alg name is unkonwn, will not dump')

    
    
    
    def _prepare_cost_calculation(self, indexes, store_size=False,unique=False):
        for index in set(indexes) - self.current_indexes:
            self._simulate_or_create_index(index, store_size=store_size,unique=unique)
        for index in self.current_indexes - set(indexes):
            self._unsimulate_or_drop_index(index,unique=unique)

        assert self.current_indexes == set(indexes)

    def _simulate_or_create_index(self, index, store_size=False,unique=False):
        if self.cost_estimation == "whatif":
            self.what_if.simulate_index(index, store_size=store_size)
        elif self.cost_estimation == "actual_runtimes":
            logging.info(f'create index {index}')
            if not unique:
                self.db_connector.create_index(index)
            else:
                self.db_connector.create_index_unique(index)
        self.current_indexes.add(index)

    
    def _unsimulate_or_drop_index(self, index,unique=False):
        if self.cost_estimation == "whatif":
            self.what_if.drop_simulated_index(index)
        elif self.cost_estimation == "actual_runtimes":
            if not unique:
                self.db_connector.drop_index(index)
            else:
                self.db_connector.drop_index_unique(index)
        self.current_indexes.remove(index)


    def _get_cost(self, query):
        
        if self.cost_estimation == "whatif":
            return self.db_connector.get_cost(query)
        elif self.cost_estimation == "actual_runtimes":
            runtime = self.db_connector.exec_query(query)[0]
            return runtime
    
    
    def complete_cost_estimation(self):
        self.completed = True

        for index in self.current_indexes.copy():
            self._unsimulate_or_drop_index(index,unique=self.have_run_calculate_cost_long)

        assert self.current_indexes == set()

    def _request_cache(self, query, indexes):
        q_i_hash = (query, frozenset(indexes))
        if q_i_hash in self.relevant_indexes_cache:
            relevant_indexes = self.relevant_indexes_cache[q_i_hash]
        else:
            relevant_indexes = self._relevant_indexes(query, indexes)
            self.relevant_indexes_cache[q_i_hash] = relevant_indexes

        
        if (query, relevant_indexes) in self.cache:
            self.cache_hits += 1
            return self.cache[(query, relevant_indexes)]
        
        else:
            if self.log_plan:
                plan=self.db_connector._get_plan(query)
                self.query_indexes_plan.append((query,indexes,plan))
                cost=plan["Total Cost"]
                self.cache[(query, relevant_indexes)] = cost
                return cost
            
            else:
                cost = self._get_cost(query)
                self.cache[(query, relevant_indexes)] = cost
                return cost


    @staticmethod
    def _relevant_indexes(query, indexes):
        relevant_indexes = [
            x for x in indexes if any(c in query.columns for c in x.columns)
        ]
        return frozenset(relevant_indexes)
