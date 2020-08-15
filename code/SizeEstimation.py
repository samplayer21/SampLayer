from typing import List

import numpy as np
from ComponentCollector import ComponentCollector
from HighGraphPreprocessing import HighGraphPreprocessing, Node
from LayerManager import LayerManager
from NeighborManager import NeighborManager
from ReachabilityEstimator import ReachabilityEstimator


class SizeEstimation:
    def __init__(self, high_subgraph: HighGraphPreprocessing,
                 neighbor_manager: NeighborManager,
                 layer_manager: LayerManager,
                 component_collector: ComponentCollector,
                 reachability_estimator: ReachabilityEstimator,
                 component_layer_num: int = 2):
        self.high_subgraph = high_subgraph
        self.neighbor_manager = neighbor_manager
        self.layer_manager = layer_manager
        self.component_collector = component_collector
        self.reachability_estimator = reachability_estimator
        self.component_layer_num = component_layer_num
        self.flush()

    def flush(self):
        self.up_nodes = []
        self.up_sum_deg = 0.
        self.up_num_nodes = 0.
        self.down_nodes = []
        self.down_weighted_sum_deg = 0.
        self.down_sum_reach = 0.

    def update_up(self, up_nodes_diff: List[Node]):
        new_up_sum_deg = np.sum([len(self.neighbor_manager.neighbors_above(v, self.component_layer_num - 1))
                                 for v in up_nodes_diff])
        self.up_nodes += up_nodes_diff
        self.up_sum_deg += new_up_sum_deg
        self.up_num_nodes += len(up_nodes_diff)

    def update_down(self, down_nodes_diff: List[Node]):
        components = [self.component_collector.component_bfs(v)
                      for v in down_nodes_diff]
        inv_reachabilities = np.array([1. / self.reachability_estimator.component_reachability(comp)
                                       for comp in components])
        degrees_down = np.array([self.component_collector.num_neighbors_of_component(comp) / len(comp)
                                 for comp in components])
        self.down_nodes += down_nodes_diff
        self.down_weighted_sum_deg += np.sum(degrees_down * inv_reachabilities)
        self.down_sum_reach += np.sum(inv_reachabilities)

    def estimate_average_deg_up(self):
        return self.up_sum_deg / self.up_num_nodes

    def estimate_average_deg_down(self):
        return self.down_weighted_sum_deg / self.down_sum_reach

    def estimate_size(self, prev_layer_size):
        up_deg = self.estimate_average_deg_up()
        down_deg = self.estimate_average_deg_down()
        return prev_layer_size * up_deg / down_deg