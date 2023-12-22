import os
import json
import pickle
import argparse
import numpy as np
from scipy.stats import ttest_ind

parser = argparse.ArgumentParser()
parser.add_argument('--metrics_dir', type=str)

# datasets = ['ptc_fr']
datasets = ['aids', 'mutag', 'ptc_fm', 'ptc_fr', 'ptc_mm', 'ptc_mr']
models = ['isonet', 'node_early_interaction', 'edge_early_interaction_vaibhav', 'edge_early_interaction_ashwin']
avg_aps_across_seeds = {key: {} for key in datasets}
t_scores = {key: {} for key in datasets}

if __name__ == "__main__":
    args = parser.parse_args()
    for file in os.listdir(args.metrics_dir):
        if file.endswith('.pkl') and file.startswith('metric'):
            print(f"Loading {file}")
            pkl = pickle.load(open(os.path.join(args.metrics_dir, file), 'rb'))
            model_name = list(pkl.keys())[0]
            surrogate_name = model_name + ("_ashwin" if "ashwin" in file.lower() else ("_vaibhav" if "vaibhav" in file.lower() else ""))
            for dataset in pkl[model_name]:
                # if dataset != 'ptc_fr': continue
                if model_name not in avg_aps_across_seeds[dataset]:
                    avg_aps_across_seeds[dataset][surrogate_name] = {}
                acc = 0
                for seed in pkl[model_name][dataset]:
                    acc += np.array(pkl[model_name][dataset][seed][1][-5])
                avg_aps_across_seeds[dataset][surrogate_name] = acc / len(pkl[model_name][dataset])

    for dataset in datasets:
        for idx_1, model_1 in enumerate(models):
            for idx_2, model_2 in enumerate(models):
                if idx_2 <= idx_1:
                    continue
                t_scores[dataset][(model_1, model_2)] = ttest_ind(
                    avg_aps_across_seeds[dataset][model_1],
                    avg_aps_across_seeds[dataset][model_2],
                    alternative='less',
                    equal_var=False
                )

    pickle.dump(t_scores, open(os.path.join(args.metrics_dir, 't_test_ap.pkl'), 'wb'))