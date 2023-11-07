import sys
import datetime
import logging
import json
import os
from selection import constants
from selection.index_selection_evaluation import IndexSelection
from index_selection_evaluation.selection.generate_matrix import CostPlanGenerate



def generate_compress_run_select():
    os.makedirs(constants.dir_path,exist_ok=True)
    cap=CostPlanGenerate(constants.config,is_load=True)
    cap.generate_query()
    cap.compress()
    cap.close()
    with open(constants.config_file) as f:
        config = json.load(f)
    
    with open('./workload_compress/result/benchmark.txt','a') as f:
        f.write(f'---{datetime.datetime.now()} start {config["benchmark_name"]}_{config["scale_factor"]}---\n')
    for com in constants.compress_algorithm:
        logging.info(f'run {com}')
        index_selection = IndexSelection(com,constants.config_file,constants.dir_path)  # pragma: no cover
        index_selection.run()
        index_selection.close()

    with open('./workload_compress/result/benchmark.txt','a') as f:
        f.write(f'---{datetime.datetime.now()} end---\n\n\n')
    logging.info(f'complete experiment \n\n\n')

generate_compress_run_select()