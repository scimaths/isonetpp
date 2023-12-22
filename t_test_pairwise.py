import os
import json
import torch
import pickle
import argparse
import numpy as np
from scipy.stats import ttest_ind

parser = argparse.ArgumentParser()
parser.add_argument('--metrics_dir', type=str)

# datasets = ['ptc_fr']
datasets = ['aids', 'mutag', 'ptc_fm', 'ptc_fr', 'ptc_mm', 'ptc_mr']
models = ['isonet', 'node_early_interaction', 'edge_early_interaction_vaibhav', 'edge_early_interaction_ashwin']
avg_pairwise_diffs = {key: {} for key in datasets}
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
                if model_name not in avg_pairwise_diffs[dataset]:
                    avg_pairwise_diffs[dataset][surrogate_name] = {}
                metrics_single = pkl[model_name][dataset][7474]
                sampler = metrics_single[1][-2]
                d_pos = sampler.list_pos
                d_neg = sampler.list_neg
                q_graphs = list(range(len(sampler.query_graphs)))
                score_diff_collection = []
                for idx, q_id in enumerate(q_graphs):
                    dpos = list(filter(lambda x:x[0][0]==q_id,d_pos))
                    dneg = list(filter(lambda x:x[0][0]==q_id,d_neg))
                    npos = len(dpos)
                    nneg = len(dneg)
                    d = dpos+dneg
                    if npos>0 and nneg>0:
                        avg = 0
                        for seed in pkl[model_name][dataset]:
                            all_preds = pkl[model_name][dataset][seed][1][-1][idx]
                            avg += (all_preds[:npos][:, None] - all_preds[npos:][None, :]).float().flatten()
                        avg /= len(pkl[model_name][dataset])
                        score_diff_collection.append(avg)
                avg_pairwise_diffs[dataset][surrogate_name] = torch.cat(score_diff_collection).numpy()

    for dataset in datasets:
        for idx_1, model_1 in enumerate(models):
            for idx_2, model_2 in enumerate(models):
                if idx_2 <= idx_1:
                    continue
                t_scores[dataset][(model_1, model_2)] = ttest_ind(
                    avg_pairwise_diffs[dataset][model_1],
                    avg_pairwise_diffs[dataset][model_2],
                    alternative='less',
                    equal_var=False
                )

    pickle.dump(t_scores, open(os.path.join(args.metrics_dir, 't_test_pairwise.pkl'), 'wb'))