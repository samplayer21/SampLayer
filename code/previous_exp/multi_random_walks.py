import argparse
import json
import igraph as ig
import logging
import numpy as np
import os
import pandas as pd
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time
from typing import NewType

path = Path(os.getcwd())
print("current path: ", path)
sys.path.insert(0, str(path / "code"))
sys.path.insert(0, str(path.parent / "code/previous_exp"))

from globals import datasets
from GraphUtils import GraphUtils as gu
from previous_exp.rejection import RejectionSampling
from previous_exp.mh import MetropolisHastingsSampling
from previous_exp.max_deg import MaxDegreeSampling

Graph = NewType("Graph", ig.GraphBase)

parser = argparse.ArgumentParser()

parser.add_argument('--output_path', default=None)
parser.add_argument('--method', choices=['rej', 'mh', 'mh+', 'md'], help='which sampling method to use: '
                                                                  'rej -- rejection sampling (Alg. 1), mh - Metropolis-Hastings, md - Max Degree',
                    default='rej')
parser.add_argument('--interval_length', type=int, default=100)
parser.add_argument('--num_intervals', type=int, default=300)
parser.add_argument('--exp_num', type=int, default=-1)
parser.add_argument('--min_exp_idx', default=0, type=int)
parser.add_argument('--cpu_count', default=0, type=int)
parser.add_argument('--min_dataset_idx', type=int)
parser.add_argument('--max_dataset_idx', type=int)
parser.add_argument('--n_steps_min', default=1, type=int)
parser.add_argument('--n_steps_max', default=1, type=int)
parser.add_argument('--n_init_nodes', default=1, type=int)
parser.add_argument('--starting_node', type=int, default=-1)

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(clientip)s %(user)-8s %(message)s')


def run_experiment(g: Graph, init_node_idx: int, experiment_index: int, method: str, title: str, total_intervals: int,
                   interval_length: int = 100, n_steps_min: int = 1, n_steps_max: int = 1):
    initial_node = g.vs[init_node_idx]
    min_deg = min(g.degree())
    max_deg = max(g.degree())
    sampler = None
    if method == 'rej':
        sampler = RejectionSampling(initial_node=initial_node, min_deg=min_deg, n_steps_min=n_steps_min, n_steps_max=n_steps_max)
    elif method == 'mh':
        sampler = MetropolisHastingsSampling(initial_node=initial_node, min_deg=min_deg, plus=False)
    elif method == 'mh+':
        sampler = MetropolisHastingsSampling(initial_node=initial_node, min_deg=min_deg, plus=True)
    elif method == 'md':
        sampler = MaxDegreeSampling(initial_node=initial_node, max_deg=max_deg)

    samples = []

    last_accepted = sampler.current_node.index
    step_counter = 0
    for _ in range(total_intervals):
        for _ in range(interval_length):
            sampler.random_step()
            node = sampler.get_node()
            node_idx = node.index if node is not None else None
            if node_idx is not None:
                last_accepted = node_idx

        step_counter += interval_length
        samples.append((step_counter, int(last_accepted), sampler.query_count))

    simulation_df = pd.DataFrame(samples, columns=['step', 'node', 'queries'])
    print(f"{title} - {experiment_index} : Finish")
    return simulation_df


def run_batch(g: Graph, dataset_idx: int, init_node_idx: int, experiment_index_range: tuple, method: str, output_path: str,
              total_intervals: int, interval_length: int, n_steps_min: int, n_steps_max: int):
    path, sep, title, directed = datasets[dataset_idx]

    start_idx, end_idx = experiment_index_range
    file_path = Path(output_path)
    print("Started batch ", experiment_index_range)
    if not file_path.exists():
        file_path.mkdir(parents=True)

    for exp_idx in range(start_idx, end_idx):
        experiment_df = run_experiment(g, init_node_idx, exp_idx, method, title,
                                       total_intervals, interval_length, n_steps_min, n_steps_max)
        experiment_df["exp"] = exp_idx
        with open(file_path / f"{title}-{method}-exp_{start_idx}-{end_idx}-init_node_{init_node_idx}.tsv", 'a') as f:
            experiment_df.to_csv(f, sep="\t", index=False, header=f.tell() == 0)
    print(f"Done with batch {experiment_index_range}")


def get_rand_nodes(graph_vcount, n_rand_nodes, seed) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice(range(graph_vcount), size=n_rand_nodes).astype(int)

if __name__ == "__main__":
    args = parser.parse_args()
    logging.info("start")
    num_cpus = args.cpu_count if args.cpu_count > 0 else cpu_count()
    pool = Pool(num_cpus)
    out_path = Path(args.output_path)
    out_path.mkdir(exist_ok=True)
    for dataset_idx in range(args.min_dataset_idx, args.max_dataset_idx + 1):
        start = time()
        dataset = datasets[dataset_idx]
        path, sep, title, directed = dataset
        g, in_degrees = gu.load_graph(path, sep, directed=directed)
        g.to_undirected()
        g = g.components().giant()
        exp_num = args.exp_num if args.exp_num > 0 else g.vcount()
        batch_size = np.ceil(exp_num / num_cpus).astype(int)
        experiment_idx_ranges = [(i, i + batch_size) for i in
                                 range(args.min_exp_idx, args.min_exp_idx + exp_num, batch_size)]
        starting_nodes = get_rand_nodes(g.vcount(), n_rand_nodes=args.n_init_nodes, seed=dataset_idx)
        if args.starting_node >= 0:
            starting_nodes = [args.starting_node,]
        params = {'title': title, 'method': args.method, 'starting_nodes': [int(v) for v in starting_nodes],
                  'min_exp_idx': args.min_exp_idx, 'exp_num': exp_num,
                  'interval_length': args.interval_length, 'num_intervals': args.num_intervals,
                  'n_steps_min': args.n_steps_min, 'n_steps_max': args.n_steps_max}

        with open(out_path / f'{title}-{args.method}-params.json', 'w') as f:
            json.dump(params, f, indent=4)

        for init_node_idx in starting_nodes:
            print(f"Starting run for {title}, starting node {init_node_idx}")
            pool.starmap(run_batch, ((g, dataset_idx, init_node_idx, experiment_idx_range, args.method, args.output_path,
                                  args.num_intervals, args.interval_length, args.n_steps_min, args.n_steps_max)
                                 for experiment_idx_range in experiment_idx_ranges))
            print(f"Finished with starting node {init_node_idx}.")

        print(f"Done with {title}. Elapsed time: {time() - start}s")
