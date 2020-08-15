import logging

import numpy as np
from ComponentCollector import ComponentCollector
from HighGraphPreprocessing import HighGraphPreprocessing
from LayerManager import LayerManager
from NeighborManager import NeighborManager, Node

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class ReachabilityEstimator:
    def __init__(self, high_subgraph: HighGraphPreprocessing,
                 neighbor_manager: NeighborManager,
                 layer_manager: LayerManager,
                 component_collector: ComponentCollector,
                 components_layer_num: int = 2):
        self.high_subgraph = high_subgraph
        self.method = self.high_subgraph.method
        self.neighbor_manager = neighbor_manager
        self.layer_manager = layer_manager
        self.component_collector = component_collector
        self.components_layer_num = components_layer_num
        self.observed_reachabilities = []

        if self.method is "greedy":
            self.node_reachability = self.node_reachability_greedy
        elif self.method is "absolute":
            self.node_reachability = self.node_reachability_absolute

    def node_reachability_greedy(self, v: Node, layer_num: int = 2, cap: int = np.inf):
        if layer_num is 0:
            return 1.

        if layer_num is 1:
            idx = self.high_subgraph.L1_dict[v.index]
            return float(self.high_subgraph.L1_degrees[idx])

        # else: layer is at least 2
        neighbors_below = self.neighbor_manager.neighbors_in_layer(v, layer_num - 1)
        if len(neighbors_below) is 0:
            return 0.

        total_reach = 0.
        for u in neighbors_below:
            reach = self.node_reachability(u, layer_num - 1)
            neighbor_contribution = reach / len(self.neighbor_manager.neighbors_above(u, layer_num - 1))
            if layer_num > 1 and total_reach + neighbor_contribution > cap:
                return np.inf
            total_reach += neighbor_contribution

        return total_reach

    def node_reachability_absolute(self, v: Node):
        assert not self.neighbor_manager.check_layer_association(v, 0) \
               and not self.neighbor_manager.check_layer_association(v, 1)
        neighbors_below = self.neighbor_manager.neighbors_in_layer(v, 1)
        return len(neighbors_below)

    def component_reachability(self, component: set, total_reach: bool = False):
        reachability_scores = [self.node_reachability(node) for node in component]
        if total_reach:
            return np.sum(reachability_scores)
        else:
            return np.mean(reachability_scores)

    def get_reachability(self, node: Node, layer_num: int):
        if layer_num is 0:
            raise ValueError("Reachability is meaningless for layer 0.")
        elif layer_num is 1:
            return 1.0
        elif layer_num >= self.components_layer_num:
            comp = self.component_collector.component_bfs(node)
            return self.component_reachability(comp)
        else:  # in the current setting (components layer is 2) this should never happen
            return self.node_reachability(node, layer_num)

    def update_observed_reachabilities(self, new_reachabilities: list):
        self.observed_reachabilities += new_reachabilities

    def estimate_baseline_reachability(self, quantile: float):
        self.observed_reachabilities.sort()
        reachabilities = self.observed_reachabilities
        min_reach = reachabilities[0]
        assert min_reach > 0
        if quantile < 0:
            quantile = 0.00001
        if quantile > 1:
            quantile = 0.99999
        weights = min_reach / np.array(reachabilities, dtype=np.float32)
        relative_weights = weights.cumsum() / weights.sum()
        above_quantile = np.flatnonzero(relative_weights >= quantile)
        try:
            return reachabilities[int(min(above_quantile))]
        except:
            return reachabilities[-1]
