import numpy as np

import random
from previous_exp.walker import Node, RandomWalker


class MetropolisHastingsSampling(RandomWalker):
    def __init__(self, initial_node: Node, min_deg: int, plus: bool = False):
        marking = not plus
        super(MetropolisHastingsSampling, self).__init__(init_node=initial_node, marking=marking)
        self.min_deg = min_deg
        self.rand_state = np.random.get_state()
        self.plus = plus

    def degree(self, node: Node):
        if not self.plus:
            self.mark(node)
        return node.degree()

    def random_step(self):
        v = self.current_node
        neighbors = self.neighbors(v)
        v_deg = len(neighbors)
        probs = [1 / max(v_deg, self.degree(u)) for u in neighbors]
        complement_prob = 1 - sum(probs)
        if complement_prob < 1e-12:
            complement_prob = 0

        probs.append(complement_prob)
        self.current_node = random.choices(neighbors + [v], weights=probs, k=1)[0]

    def get_node(self):
        return self.current_node
