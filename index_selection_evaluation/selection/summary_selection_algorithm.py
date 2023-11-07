import logging
from time import time

from .summary_cost_evaluation import CostEvaluation

from .ProcessedWorkloadSummary import ProcessedWorkloadSummary as WorkloadSummary



DEFAULT_PARAMETER_VALUES = {
    "budget_MB": 500,
    "max_indexes": 15,
    "max_index_width": 2,
}

class SummarySelectionAlgorithm:
    def __init__(self, database_connector, parameters, default_parameters=None):
        if default_parameters is None:
            default_parameters = {}
        logging.debug("Init selection algorithm")
        self.did_run = False
        self.parameters = parameters
        
        for key, value in default_parameters.items():
            if key not in self.parameters:
                self.parameters[key] = value

        self.database_connector = database_connector
        self.database_connector.drop_indexes()
        self.name='unset'
        self.full_workload_path=None

        self.cost_evaluation = CostEvaluation(database_connector,full_workload_path=self.full_workload_path,log_plan=parameters['log_plan'])
        
        if "cost_estimation" in self.parameters:
            estimation = self.parameters["cost_estimation"]
            self.cost_evaluation.cost_estimation = estimation
        
        self.relevant_indexes_cache = {}

        self.cache = {}
    
    def set_compress_name(self,name):
        self.cost_evaluation.set_compress_name(name)

    def calculate_best_indexes(self, workload):
        assert self.did_run is False, "Selection algorithm can only run once."
        self.cost_evaluation.set_alg_name(self.name)
        self.did_run = True
        self.ws=WorkloadSummary(workload,self.cost_evaluation)
        self.summaried_workload=self.ws.get_summary_workload()
        self.query2i_dict={}
        self.i2query_dict={}
        for i,query in enumerate(self.summaried_workload.queries):
            self.query2i_dict[query]=i
            self.i2query_dict[i]=query
        self.summary_workload_list=[i for i in range(len(self.summaried_workload.queries))]
        indexes = self._calculate_best_indexes(self.summaried_workload)
        self._log_cache_hits()
        
        self.cost_evaluation.complete_cost_estimation()

        return indexes

    def _calculate_best_indexes(self, workload):
        raise NotImplementedError("_calculate_best_indexes(self, " "workload) missing")

    def _log_cache_hits(self):
        
        return
        

    def _request_cache(self, query, indexes):
        q_i_hash = (query, frozenset(indexes))
        if q_i_hash in self.relevant_indexes_cache:
            relevant_indexes = self.relevant_indexes_cache[q_i_hash]
        else:
            relevant_indexes = self._relevant_indexes(query, indexes)
            self.relevant_indexes_cache[q_i_hash] = relevant_indexes

        
        if (query, relevant_indexes) in self.cache:
            return self.cache[(query, relevant_indexes)],None
        else:
            return -1,relevant_indexes

    @staticmethod
    def _relevant_indexes(query, indexes):
        relevant_indexes = [
            x for x in indexes if any(c in query.columns for c in x.columns)
        ]
        return frozenset(relevant_indexes)


class NoIndexAlgorithm(SummarySelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SummarySelectionAlgorithm.__init__(self, database_connector, parameters)
        self.name='noIndex'

    def _calculate_best_indexes(self, workload):
        return []


class AllIndexesAlgorithm(SummarySelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SummarySelectionAlgorithm.__init__(self, database_connector, parameters)
        self.name='allIndex'

    
    def _calculate_best_indexes(self, workload):
        return workload.potential_indexes()
