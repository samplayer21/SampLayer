## Dependencies
The following packages are required:

```
autologging
joblib
matplotlib
numpy
pandas
python-igraph
tqdm
```

## General Notes
The list of relative network paths, column separators, titles, and directed flags (whether they are directed or not) are listed in the script `code/global.py`. The file is used by both our algorithms and the previous algorithm implementations.

## Running SampLayer and SampLayer+
Our code provides two algorithms, SampLayer and SampLayer+, for sampling random nodes. 
SampLayer only assumes standard query access (querying a node reveals list of neighbors).
SampLayer+ also assumes query returns degrees of neighbors.

Usage
```
python code/SampLayer.py [args]
```

where args:

- `--output_path` Where to dump the results.
- `--num_samples` number of node samples to generate
- `--plus` adding this flag runs SampLayer+ (default is SampLayer).
- `--dataset_idx` network index (according to the list in `code/globals.py`).
- `--eps` target distance from uniformity
- `--L0_size` number of nodes to query for L0. Optional; Defaults to values mentioned in paper.
- `--L1_num_initial_samples` number of nodes from L1 used for L2-size estimation. Optional; Defaults to values mentioned in paper.
- `--L2_num_initial_reaches` number of nodes reached in L2 used for L2-size estimation. Optional; Defaults to values mentioned in paper.
- `--num_additional_L1_nodes_per_sample` optional; additional node sampling from L1 to improve size estimation for L2.

Example:
`python code/SampLayer.py --num_samples 100 --eps 0.03 --dataset_idx 0 --L0_size 2000 --L1_num_initial_samples 1000 --L2_num_initial_reaches 150 --num_additional_L1_nodes_per_sample 2 --output_path foo.tsv`

## Previous Random Walk Algorithms
Our results are benchmarked against the random walk algorithms `Rej`, `MH`, and `MH+` from the paper [On Sampling Nodes in a Network](https://dl.acm.org/doi/10.1145/2872427.2883045) by Chiericetti et al.
The algorithm `MD` from their paper was not tested due to impractical running time.

Usage:
```
python code/previous_exp/multi_random_walks.py [args]
```
where args:

- `--output_path` Where to dump the results.
- `--method {rej,mh,mh+,md}` : which sampling method to use: `rej` -- rejection sampling, `mh` -- Metropolis-Hastings, `mh+` -- Metropolis-Hastings with the stronger query model, `md` -- maximum degree.
- `--interval_length` Number of steps before each sample.
- `--num_intervals` How many intervals in a single walk.
- `--exp_num` Number of walks.
- `--min_exp_idx` Minimum experiment index to start the count from.
- `--cpu_count` Number of cores to use (default: number of cores).
- `--min_dataset_idx` Minimum network index (according to the list in `code/globals.py`).
- `--max_dataset_idx` Max of above.
- `--n_steps_min` Relevant to `rej` only.
- `--n_steps_max` Relevant to `rej` only.
- `--n_init_nodes` Number of initial nodes at the start of random walks (there are `exp_num` such walks).
- `--starting_node` Optional: specify initial node.

Example:
`python code/previous_exp/multi_random_walks.py --method rej --output_path bla --interval_length 10 --num_intervals 100 --exp_num 100 --cpu_count 4 --min_dataset_idx 0 --max_dataset_idx 0 --n_steps_min 5 --n_steps_max 9 --n_init_nodes 1 --starting_node 100`
