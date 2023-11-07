import logging

from ..index import Index
from ..selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm
from ..utils import b_to_mb, mb_to_b
import time



DEFAULT_PARAMETERS = {
    "budget_MB": DEFAULT_PARAMETER_VALUES["budget_MB"],
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
    "min_cost_improvement": 1.003,
}


class ExtendAlgorithm(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(
            self, database_connector, parameters, DEFAULT_PARAMETERS
        )
        self.budget = mb_to_b(self.parameters["budget_MB"])
        self.max_index_width = self.parameters["max_index_width"]
        self.workload = None
        self.min_cost_improvement = self.parameters["min_cost_improvement"]
        self.index_size_time=0
        self.processed_index=set()

    def _calculate_best_indexes(self, workload):
        logging.info("Calculating best indexes Extend")
        self.workload = workload
        single_attribute_index_candidates = self.workload.potential_indexes()
        extension_attribute_candidates = single_attribute_index_candidates.copy()

        
        index_combination = []
        index_combination_size = 0
        
        best = {"combination": [], "benefit_to_size_ratio": 0, "cost": None}
        self._set_indexes_size(index_combination)
        current_cost = self.cost_evaluation.calculate_cost(
            self.workload, index_combination, store_size=True
        )
        self.initial_cost = current_cost
        
        while True:
            single_attribute_index_candidates = self._get_candidates_within_budget(
                index_combination_size, single_attribute_index_candidates
            )
            for candidate in single_attribute_index_candidates:
                
                if candidate not in index_combination:
                    self._evaluate_combination(
                        index_combination + [candidate], best, current_cost
                    )

            for attribute in extension_attribute_candidates:
                
                
                self._attach_to_indexes(index_combination, attribute, best, current_cost)
            if best["benefit_to_size_ratio"] <= 0:
                break

            index_combination = best["combination"]
            index_combination_size = sum(
                index.estimated_size for index in index_combination
            )
            logging.debug(
                "Add index. Current cost savings: "
                f"{(1 - best['cost'] / current_cost) * 100:.3f}, "
                f"initial {(1 - best['cost'] / self.initial_cost) * 100:.3f}. "
                f"Current storage: {index_combination_size:.2f}"
            )

            best["benefit_to_size_ratio"] = 0
            current_cost = best["cost"]
        logging.info(f'构建Index size的时间: {self.index_size_time}')
        return index_combination

    def _attach_to_indexes(self, index_combination, attribute, best, current_cost):
        assert (
            attribute.is_single_column() is True
        ), "Attach to indexes called with multi column index"

        for position, index in enumerate(index_combination):
            if len(index.columns) >= self.max_index_width:
                continue
            if index.appendable_by(attribute):
                new_index = Index(index.columns + attribute.columns)
                if new_index in index_combination:
                    continue
                new_combination = index_combination.copy()
                
                del new_combination[position]
                new_combination.append(new_index)
                self._evaluate_combination(
                    new_combination,
                    best,
                    current_cost,
                    index_combination[position].estimated_size,
                )

    def _get_candidates_within_budget(self, index_combination_size, candidates):
        new_candidates = []
        for candidate in candidates:
            if (candidate.estimated_size is None) or (
                candidate.estimated_size + index_combination_size <= self.budget
            ):
                new_candidates.append(candidate)
        return new_candidates

    def _evaluate_combination(
        self, index_combination, best, current_cost, old_index_size=0
    ):
        self._set_indexes_size(index_combination)
        cost = self.cost_evaluation.calculate_cost(
            self.workload, index_combination, store_size=True
        )
        if (cost * self.min_cost_improvement) >= current_cost:
            return
        benefit = current_cost - cost
        new_index = index_combination[-1]
        new_index_size_difference = new_index.estimated_size - old_index_size
        
        if b_to_mb(new_index_size_difference) < 0.1:
            return
        assert new_index_size_difference != 0, "Index size difference should not be 0!"

        ratio = benefit / new_index_size_difference

        total_size = sum(index.estimated_size for index in index_combination)

        if ratio > best["benefit_to_size_ratio"] and total_size <= self.budget:
            logging.debug(
                f"new best cost and size: {cost}\t" f"{b_to_mb(total_size):.2f}MB"
            )
            best["combination"] = index_combination
            best["benefit_to_size_ratio"] = ratio
            best["cost"] = cost
    
    
    def _set_indexes_size(self,index_combination):
        start=time.time()
        for index in index_combination:
            if index in self.processed_index:
                continue
            self.cost_evaluation.what_if.simulate_index(index,store_size=True)
            self.cost_evaluation.what_if.drop_simulated_index(index)
            self.processed_index.add(index)
        end=time.time()
        self.index_size_time=end-start
        