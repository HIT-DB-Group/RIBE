import logging

from .cost_evaluation import CostEvaluation
from .utils import load_checkpoint,save2checkpoint
import os



DEFAULT_PARAMETER_VALUES = {
    "budget_MB": 500,
    "max_indexes": 15,
    "max_index_width": 2,
}

class SelectionAlgorithm:
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
        if 'full_workload_path' in parameters:
            self.full_workload_path=parameters['full_workload_path']
        
        self.cost_evaluation = CostEvaluation(database_connector,full_workload_path=self.full_workload_path,log_plan=parameters['log_plan'])
        
        if "cost_estimation" in self.parameters:
            estimation = self.parameters["cost_estimation"]
            self.cost_evaluation.cost_estimation = estimation
    def set_compress_name(self,name):
        self.cost_evaluation.set_compress_name(name)
    def calculate_best_indexes(self, workload):
        assert self.did_run is False, "Selection algorithm can only run once."
        self.cost_evaluation.set_alg_name(self.name)
        self.did_run = True
        indexes = self._calculate_best_indexes(workload)
        self._log_cache_hits()
        
        self.cost_evaluation.complete_cost_estimation()

        return indexes

    def _calculate_best_indexes(self, workload):
        raise NotImplementedError("_calculate_best_indexes(self, " "workload) missing")

    def _log_cache_hits(self):
        
        hits = self.cost_evaluation.cache_hits
        requests = self.cost_evaluation.cost_requests
        logging.debug(f"Total cost cache hits:\t{hits}")
        logging.debug(f"Total cost requests:\t\t{requests}")
        if requests == 0:
            return
        ratio = round(hits * 100 / requests, 2)
        logging.debug(f"Cost cache hit ratio:\t{ratio}%")


class NoIndexAlgorithm(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(self, database_connector, parameters)
        self.name='noIndex'

    def _calculate_best_indexes(self, workload):
        return []


class AllIndexesAlgorithm(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(self, database_connector, parameters)
        self.name='allIndex'

    
    def _calculate_best_indexes(self, workload):
        return workload.potential_indexes()
