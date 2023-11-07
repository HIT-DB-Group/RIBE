import copy
import json
import logging
import pickle
import sys
import time
import os

from .algorithms.anytime_algorithm import AnytimeAlgorithm
from .algorithms.auto_admin_algorithm import AutoAdminAlgorithm
from .algorithms.db2advis_algorithm import DB2AdvisAlgorithm
from .algorithms.dexter_algorithm import DexterAlgorithm
from .algorithms.drop_heuristic_algorithm import DropHeuristicAlgorithm
from .algorithms.extend_algorithm import ExtendAlgorithm
from .algorithms.relaxation_algorithm import RelaxationAlgorithm
from .algorithms.summary_anytime import SummaryAnytimeAlgorithm

from .benchmark import Benchmark

from .dbms.hana_dbms import HanaDatabaseConnector
from .dbms.postgres_dbms import PostgresDatabaseConnector
from .query_generator import QueryGenerator
from .selection_algorithm import AllIndexesAlgorithm, NoIndexAlgorithm
from .table_generator import TableGenerator
from .workload import Workload
from .utils import save2checkpoint,load_checkpoint

ALGORITHMS = {
    "anytime": AnytimeAlgorithm,
    "auto_admin": AutoAdminAlgorithm,
    "db2advis": DB2AdvisAlgorithm,
    "dexter": DexterAlgorithm,
    "drop": DropHeuristicAlgorithm,
    "extend": ExtendAlgorithm,
    "relaxation": RelaxationAlgorithm,
    "no_index": NoIndexAlgorithm,
    "all_indexes": AllIndexesAlgorithm,
    "summary_anytime":SummaryAnytimeAlgorithm
}

DBMSYSTEMS = {"postgres": PostgresDatabaseConnector, "hana": HanaDatabaseConnector}


class IndexSelection:
    def __init__(self,compress_algorithm=None,default_config_file=None,dir_path=None):
        logging.debug("Init IndexSelection")
        self.db_connector = None
        if default_config_file is not None:
            self.default_config_file = default_config_file
        else:
            self.default_config_file = "example_configs/config_tpch.json"
            
        self.disable_output_files = False
        self.database_name = None
        self.database_system = None
        self.compress_algorithm=compress_algorithm
        self.compress_algorithms_filter=None
        self.dir_path=dir_path
        

    def run(self):
        """This is called when running `python3 -m selection`.
        """
        config_file = self._parse_command_line_args()
        if not config_file:
            config_file = self.default_config_file

        logging.info("Starting Index Selection Evaluation")
        logging.info("Using config file {}".format(config_file))

        self._run_algorithms(config_file)

    def _setup_config(self, config):
        dbms_class = DBMSYSTEMS[config["database_system"]]
        generating_connector = dbms_class(None, autocommit=True)
        self.benchmark_name=config["benchmark_name"]
        
        
        table_generator = TableGenerator(
            config["benchmark_name"], config["scale_factor"], generating_connector
        )
        self.database_name = table_generator.database_name()
        self.database_system = config["database_system"]
        self.setup_db_connector(self.database_name, self.database_system)

        self.full_workload_path=None
        if 'full_workload_path' in config:
            self.full_workload_path=self.dir_path+'/'+config['full_workload_path']

        if "queries" not in config:
            config["queries"] = None
        query_generator = QueryGenerator(
            config["benchmark_name"],
            config["scale_factor"],
            self.db_connector,
            config["queries"],
            table_generator.columns,
        )

        if "compress_algorithms_filter" in config:
            self.compress_algorithms_filter=config["compress_algorithms_filter"]
            
        
        if 'compress_workload_path' in config and (self.compress_algorithm is not None):
            compress_workload_path=self.dir_path+'/'+config['compress_workload_path']
            with open(f'{compress_workload_path}_{self.compress_algorithm}','rb') as f:
                self.workload = pickle.load(f)
            print('load compressed workload')
        else:
            self.workload = Workload(query_generator.queries)
            print('generate workload')

        if "pickle_workload" in config and config["pickle_workload"] is True:
            pickle_filename = (
                f"benchmark_results/workload_{config['benchmark_name']}"
                f"_{len(self.workload.queries)}_queries.pickle"
            )
            pickle.dump(self.workload, open(pickle_filename, "wb"))

    def _run_algorithms(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        self._setup_config(config)
        self.db_connector.drop_indexes()

        
        
        self.db_connector.create_statistics()
        self.db_connector.commit()

        
        for algorithm_config in config["algorithms"]:
            
            
            if algorithm_config["name"] == "cophy":
                continue
            if self.compress_algorithms_filter is not None:
                if self.compress_algorithm in self.compress_algorithms_filter.keys():
                    if algorithm_config["name"] in self.compress_algorithms_filter[self.compress_algorithm]:
                        continue
            
            
            
            configs = self._find_parameter_list(algorithm_config)
            
            for algorithm_config_unfolded in configs:
                cfg = algorithm_config_unfolded
                
                if "max_indexes" in cfg["parameters"]:
                    cfg_dir_path=f'{self.dir_path}/max_indexes{cfg["parameters"]["max_indexes"]}_max_index_width{cfg["parameters"]["max_index_width"]}'
                elif "budget_MB" in cfg["parameters"]:
                    cfg_dir_path=f'{self.dir_path}/budget_MB{cfg["parameters"]["budget_MB"]}_max_index_width{cfg["parameters"]["max_index_width"]}'
                else:
                    cfg_dir_path=f'{self.dir_path}/max_index_width{cfg["parameters"]["max_index_width"]}'
                os.makedirs(cfg_dir_path,exist_ok=True)
        
                start_time = time.time()

                indexes, what_if, cost_requests, cache_hits = self._run_algorithm(cfg,cfg_dir_path)
                calculation_time = round(time.time() - start_time, 2)
                with open('./workload_compress/result/benchmark.txt','a') as f:
                    f.write(f'time consume, {calculation_time},')

                benchmark = Benchmark(
                    self.workload,
                    indexes,
                    self.db_connector,
                    algorithm_config_unfolded,
                    calculation_time,
                    self.disable_output_files,
                    config,
                    cost_requests,
                    cache_hits,
                    what_if,
                    algorithm_config["name"],
                    self.compress_algorithm,
                    self.full_workload_path,
                    cfg_dir_path
                )
                benchmark.benchmark()
                if self.dir_path is not None:
                    with open('./workload_compress/result/benchmark.txt','r') as f:
                        text=f.read().split('\n')[-2]
                    with open(f'{cfg_dir_path}/result.csv','a') as f:
                        f.write(text)
                        f.write('\n')

    
    
    def _find_parameter_list(self, algorithm_config):
        parameters = algorithm_config["parameters"]
        configs = []
        if parameters:
            
            self.__check_parameters(parameters)
            for key, value in parameters.items():
                if isinstance(value, list):
                    for i in value:
                        new_config = copy.deepcopy(algorithm_config)
                        new_config["parameters"][key] = i
                        configs.append(new_config)
        if len(configs) == 0:
            configs.append(algorithm_config)
        return configs

    def __check_parameters(self, parameters):
        counter = 0
        for key, value in parameters.items():
            if isinstance(value, list):
                counter += 1
        if counter > 1:
            raise Exception("Too many parameter lists in config")

    def _run_algorithm(self, config,cfg_dir_path=None):
        self.db_connector.drop_indexes()
        self.db_connector.commit()
        self.setup_db_connector(self.database_name, self.database_system)

        algorithm = self.create_algorithm_object(config["name"], config["parameters"])
        logging.info(f"Running algorithm {config}")
        if self.compress_algorithm is not None:
            algorithm.set_compress_name(self.compress_algorithm)
        
        
        indexes = algorithm.calculate_best_indexes(self.workload)

        logging.info(f"Indexes found: {indexes}")
        
        what_if = algorithm.cost_evaluation.what_if

        cost_requests = (
            self.db_connector.cost_estimations
            if config["name"] == "db2advis"
            else algorithm.cost_evaluation.cost_requests
        )
        cache_hits = (
            0 if config["name"] == "db2advis" else algorithm.cost_evaluation.cache_hits
        )
        return indexes, what_if, cost_requests, cache_hits

    def create_algorithm_object(self, algorithm_name, parameters):
        if self.full_workload_path is not None:
            parameters['full_workload_path']=self.full_workload_path
        algorithm = ALGORITHMS[algorithm_name](self.db_connector, parameters)
        return algorithm

    def _parse_command_line_args(self):
        arguments = sys.argv
        if "CRITICAL_LOG" in arguments:
            logging.getLogger().setLevel(logging.CRITICAL)
        if "ERROR_LOG" in arguments:
            logging.getLogger().setLevel(logging.ERROR)
        if "INFO_LOG" in arguments:
            logging.getLogger().setLevel(logging.INFO)
        if "DISABLE_OUTPUT_FILES" in arguments:
            self.disable_output_files = True
        for argument in arguments:
            if ".json" in argument:
                return argument

    def setup_db_connector(self, database_name, database_system):
        if self.db_connector:
            logging.info("Create new database connector (closing old)")
            self.db_connector.close()
        self.db_connector = DBMSYSTEMS[database_system](database_name)
    def close(self):
        self.db_connector.commit()
        self.db_connector.close()