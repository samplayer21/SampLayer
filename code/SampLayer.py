import argparse

import pandas as pd

from Sampler import Sampler
from globals import nums_L0, nums_L0_plus, nums_L1_up, nums_L1_up_plus, nums_L2_reaches, nums_L2_reaches_plus

parser = argparse.ArgumentParser()

parser.add_argument('--output_path', default=None)
parser.add_argument('--plus', action='store_true', help="use SampLayer+ ? otherwise (by default) uses SampLayer")
parser.add_argument('--dataset_idx', type=int, help="see globals.py")
parser.add_argument('--eps', type=float, default=0.05, help="target distance from uniformity")
parser.add_argument('--L0_size', type=int, default=-1, help="Number of nodes queried in L0")
parser.add_argument('--L1_num_initial_samples', type=int, default=-1, help="Number of nodes from L1 for L2-size estimation")
parser.add_argument('--L2_num_initial_reaches', type=int, default=-1, help="Number of reached nodes in L2 for L2-size estimation")
parser.add_argument('--num_samples', type=int, help="Number of node samples to generate")
parser.add_argument('--num_additional_L1_nodes_per_sample', type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()

    # preprocessing
    sampler = Sampler(dataset_idx=args.dataset_idx, plus=args.plus)
    dataset_name = sampler.title

    L0_size = args.L0_size if args.L0_size > 0 else (nums_L0_plus[dataset_name] if args.plus else nums_L0[dataset_name])
    L1_num_initial_samples = args.L1_num_initial_samples if args.L1_num_initial_samples > 0 else \
        (nums_L1_up_plus[dataset_name] if args.plus else nums_L1_up[dataset_name])
    L2_num_initial_reaches = args.L2_num_initial_reaches if args.L2_num_initial_reaches > 0 else \
        (nums_L2_reaches_plus[dataset_name] if args.plus else nums_L2_reaches[dataset_name])

    sampler.generate_L0(L0_size=L0_size)
    sampler.preprocess_L2(L1_num_samples=L1_num_initial_samples,
                          L2_num_reaches=L2_num_initial_reaches,
                          allowed_error=args.eps)

    # sampling
    node_samples, query_counts = sampler.sample(num_samples=args.num_samples,
                                                num_additional_L1_samples=args.num_additional_L1_nodes_per_sample)
       
    indices = [str(node.index) for node in node_samples]
    results_df = pd.DataFrame(data={'node': indices, 'queries': query_counts})
    if args.output_path is None:
        print("Sampled nodes:")
        print(results_df)
    else:
        results_df.to_csv(args.output_path, sep='\t', index=False)
