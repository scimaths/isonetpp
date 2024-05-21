import os
import json
import torch
from itertools import product
from utils.tooling import read_config
from utils.experiment import Experiment
from subgraph_matching.test import evaluate_model
from subgraph_matching.model_handler import get_model
from subgraph_matching.dataset import SubgraphIsomorphismDataset

ITER_COUNT = 1

def modify_and_evaluate_model(
    model: torch.nn.Module, dataset, modifications: dict
):
    attribute_archive = {}
    for key, value in modifications.items():
        attribute_archive[key] = model.__dict__[key]
        model.__dict__[key] = value

    ap_score, map_score, avg_running_time = evaluate_model(model, dataset, return_running_time=True)

    for key, value in attribute_archive.items():
        model.__dict__[key] = attribute_archive[key]

    return map_score, avg_running_time

def load_models(paths, max_node_set_size, max_edge_set_size, device):
    name_to_model_dict = {}
    for model_path in paths:
        dumped_model_config = json.load(open(
            model_path.replace("trained_models", "configs").replace("pth", "json"), 'r'
        ))
        model_name = dumped_model_config['model']
        experiment_id = dumped_model_config['experiment_id']
        if "edge_early" in model_name:
            complete_config = read_config(os.path.join("configs", f"{model_name}.yaml"))
        else:
            complete_config = read_config(os.path.join(
                "configs",
                experiment_id,
                f"{model_name.replace('gmn_iterative_refinement_', 'iterative___').replace('gmn_baseline_', '')}.yaml")
            )

        model = get_model(
            model_name, config=complete_config.model_config, max_node_set_size=max_node_set_size,
            max_edge_set_size=max_edge_set_size, device=device
        )
        with open(model_path, 'rb') as model_handler:
            model_state_dict = torch.load(model_handler)['model_state_dict']
        model.load_state_dict(model_state_dict)
        model.to(device)

        name_to_model_dict[model_name] = model

    return name_to_model_dict

if __name__ == "__main__":
    test_dataset_dict = {}
    for dataset_name in ["aids", "mutag", "ptc_fm", "ptc_fr", "ptc_mm", "ptc_mr"]:
        test_dataset_dict[dataset_name] = SubgraphIsomorphismDataset(
            mode="test", dataset_name=dataset_name, dataset_size="large",
            batch_size=128, data_type="gmn", dataset_base_path=".", experiment=None
        )

    modifications = {
        'gmn_baseline': {
            'propagation_steps': range(3, 16),
        },
        'gmn_iterative_refinement': {
            'propagation_steps': range(3, 9),
            'refinement_steps': range(3, 6),
        },
        # 'edge_early_interaction': {
        #     'propagation_steps': range(3, 9),
        #     'time_update_steps': range(3, 6),
        # }
    }

    model_paths = {
        'aids': [
            # 'efficiency_experiments/trained_models/gmn_baseline_scoring=agg___tp=attention_pp=identity_when=post_aids_large_dataset_seed_7366_2024-04-18_01:23:36.pth',
            'efficiency_experiments/trained_models/gmn_baseline_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true_aids_large_dataset_seed_7762_2024-04-18_14:19:00.pth',
            'efficiency_experiments/trained_models/gmn_iterative_refinement_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true_aids_large_dataset_seed_7762_2024-04-02_14:00:34.pth',
        ],
        'mutag': [
            # 'efficiency_experiments/trained_models/gmn_baseline_scoring=agg___tp=attention_pp=identity_when=post_mutag_large_dataset_seed_4929_2024-04-02_13:58:29.pth',
            'efficiency_experiments/trained_models/gmn_baseline_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true_mutag_large_dataset_seed_7762_2024-04-18_14:18:10.pth',
            'efficiency_experiments/trained_models/gmn_iterative_refinement_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true_mutag_large_dataset_seed_7762_2024-04-02_13:59:44.pth',
        ],
        'ptc_fm': [
            # 'efficiency_experiments/trained_models/gmn_baseline_scoring=agg___tp=attention_pp=identity_when=post_ptc_fm_large_dataset_seed_7366_2024-04-02_13:59:05.pth',
            'efficiency_experiments/trained_models/gmn_baseline_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true_ptc_fm_large_dataset_seed_7474_2024-04-18_01:25:26.pth',
            'efficiency_experiments/trained_models/gmn_iterative_refinement_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true_ptc_fm_large_dataset_seed_7474_2024-04-01_23:49:26.pth',
        ],
        'ptc_fr': [
            # 'efficiency_experiments/trained_models/gmn_baseline_scoring=agg___tp=attention_pp=identity_when=post_ptc_fr_large_dataset_seed_7474_2024-04-18_01:23:16.pth',
            'efficiency_experiments/trained_models/gmn_baseline_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true_ptc_fr_large_dataset_seed_7762_2024-04-18_14:18:40.pth',
            'efficiency_experiments/trained_models/gmn_iterative_refinement_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true_ptc_fr_large_dataset_seed_7762_2024-04-02_14:00:14.pth',
        ],
        'ptc_mm': [
            # 'efficiency_experiments/trained_models/gmn_baseline_scoring=agg___tp=attention_pp=identity_when=post_ptc_mm_large_dataset_seed_7474_2024-04-18_14:23:55.pth',
            'efficiency_experiments/trained_models/gmn_baseline_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true_ptc_mm_large_dataset_seed_7762_2024-02-29_00:55:10.pth',
            'efficiency_experiments/trained_models/gmn_iterative_refinement_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true_ptc_mm_large_dataset_seed_7762_2024-03-24_16:09:04.pth',
        ],
        'ptc_mr': [
            # 'efficiency_experiments/trained_models/gmn_baseline_scoring=agg___tp=attention_pp=identity_when=post_ptc_mr_large_dataset_seed_7366_2024-03-28_01:21:03.pth',
            'efficiency_experiments/trained_models/gmn_baseline_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true_ptc_mr_large_dataset_seed_7366_2024-02-29_00:54:39.pth',
            'efficiency_experiments/trained_models/gmn_iterative_refinement_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true_ptc_mr_large_dataset_seed_7366_2024-03-24_16:08:54.pth',
        ],
    }

    all_map_scores = {}

    for dataset_name, test_dataset in test_dataset_dict.items():
        all_map_scores[dataset_name] = {}
        name_to_model_dict = load_models(
            paths=model_paths[dataset_name],
            max_node_set_size=test_dataset.max_node_set_size,
            max_edge_set_size=test_dataset.max_edge_set_size,
            device=test_dataset.device
        )
        for model_name, model in name_to_model_dict.items():
            all_map_scores[dataset_name][model_name] = {}
            if "gmn_baseline" in model_name:
                for propagation_steps in range(3, 8):
                    all_running_times = []
                    for iter in range(ITER_COUNT):
                        map_score, avg_running_time = modify_and_evaluate_model(
                            model, test_dataset,
                            modifications={'propagation_steps': propagation_steps}
                        )
                        all_running_times.append(avg_running_time)
                        print(dataset_name, model_name, f"K={propagation_steps}", map_score, avg_running_time)
                    all_map_scores[dataset_name][model_name][f"({propagation_steps})"] = (map_score, all_running_times)
                    json.dump(all_map_scores, open('metric_vs_efficiency.json', 'w'))
            elif "gmn_iterative_refinement" in model_name:
                for refinement_steps, propagation_steps in [(3, 5), (3, 10), (4, 5), (5, 5)]:
                    all_running_times = []
                    for iter in range(ITER_COUNT):
                        map_score, avg_running_time = modify_and_evaluate_model(
                            model, test_dataset,
                            modifications={
                                'propagation_steps': propagation_steps,
                                'refinement_steps': refinement_steps,
                            }
                        )
                        all_running_times.append(avg_running_time)
                        print(dataset_name, model_name, f"T={refinement_steps}, K={propagation_steps}", map_score, avg_running_time)
                    all_map_scores[dataset_name][model_name][f"({refinement_steps},{propagation_steps})"] = (map_score, all_running_times)
                    json.dump(all_map_scores, open('metric_vs_efficiency.json', 'w'))
