import os

import torch
import numpy as np
import pickle
import json
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

from subgraph_matching import dataset
from subgraph_matching.dataset import SubgraphIsomorphismDataset
from subgraph_matching.model_handler import get_model
from utils.tooling import read_config


def get_models(
    paths_to_experiment_dir = [
        "/mnt/nas/vaibhavraj/isonetpp_experiments/",
        "/mnt/nas/vaibhavraj/isonetpp_experiments_march_16/",
        "/mnt/nas/vaibhavraj/isonet_experiments_02_april/",
        "/mnt/home/vaibhavraj/isonetpp_enhanced_code/experiments/",
        "/mnt/home/vaibhavraj/isonetpp_enhanced_code/experiments_archived_march_16/",
        "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/experiments_updated/",
    ]
):
    for experiment_dir in paths_to_experiment_dir:
        for experiment in os.listdir(experiment_dir):

            model_paths = []
            for model in sorted(os.listdir(experiment_dir + experiment + "/trained_models")):
                model_paths.append(experiment_dir + experiment + "/trained_models/" + model)

            config_paths = []
            for config in sorted(os.listdir(experiment_dir + experiment + "/configs")):
                config_paths.append(experiment_dir + experiment + "/configs/" + config)

            model_config_pair = []
            for model_path in model_paths:
                model_name = model_path.split("/")[-1].split(".")[0]

                index = 0
                for config_path in config_paths:
                    if model_name in config_path:
                        index += 1
                        model_config_pair.append((model_path, config_path))

                if index > 1:
                    raise ValueError("More than one config file for model")
                if index == 0:
                    raise ValueError("No config file for model")

            for model_path, config_path in model_config_pair:
                yield model_path, config_path

def load_config():
    model_name_to_config_map = {}
    for root, _, files in os.walk("./configs/"):
        for file in files:
            config_path = os.path.join(root, file)
            model_params, _ = read_config(config_path, with_dict=True)
            if 'model_config' in model_params and 'name' in model_params:
                model_name_to_config_map[model_params.name] = config_path
    return model_name_to_config_map


def load_datasets(data_type):
    dataset_map = {}
    for dataset_name in ["aids", "mutag", "ptc_fm", "ptc_fr", "ptc_mm", "ptc_mr"]:
        dataset_map[dataset_name] = SubgraphIsomorphismDataset(
            mode = 'test',
            dataset_name = dataset_name,
            dataset_size = "large",
            batch_size = 128,
            data_type = data_type,
            dataset_base_path = ".",
            experiment = None
        )
    return dataset_map



def eval_node_alignment(
    model,
    dataset,
    max_node_set_size
):
    model.eval()
    pos_pairs, neg_pairs = dataset.pos_pairs, dataset.neg_pairs
    num_query_graphs = len(dataset.query_graphs)

    positive_histogram = []
    negative_histogram = []

    for query_idx in range(num_query_graphs):
        pos_pairs_for_query = list(filter(lambda pair: pair[0] == query_idx, pos_pairs))
        neg_pairs_for_query = list(filter(lambda pair: pair[0] == query_idx, neg_pairs))

        query_adjacency_matrix = []
        corpus_adjacency_matrix = []

        if len(pos_pairs_for_query) > 0 and len(neg_pairs_for_query) > 0:
            all_pairs = pos_pairs_for_query + neg_pairs_for_query
            num_pos_pairs, num_neg_pairs = len(pos_pairs_for_query), len(neg_pairs_for_query)

            soft_permutations = []
            num_batches = dataset.create_custom_batches(all_pairs)
            for batch_idx in range(num_batches):
                batch_graphs, batch_graph_sizes, _, batch_adj_matrices = dataset.fetch_batch_by_id(batch_idx)
                soft_permutations.append(
                    # TODO: fix [0]
                    model.forward_for_alignment(batch_graphs, batch_graph_sizes, batch_adj_matrices)[0]
                )
                query_adjacency_matrix.extend([matrices[0] for matrices in batch_adj_matrices])
                corpus_adjacency_matrix.extend([matrices[1] for matrices in batch_adj_matrices])

            # Get hard permutation matrix
            soft_permutations = torch.cat(soft_permutations, dim=0)
            hard_permutation = []
            for soft_perm in soft_permutations:
                soft_perm = soft_perm.detach().cpu()
                row_ind, col_ind = linear_sum_assignment(-soft_perm)
                hard_permutation.append(torch.eye(max_node_set_size, device=model.device)[col_ind])
            hard_permutation = torch.stack(hard_permutation)

            # Calculate hinge score
            query_adjacency_matrix = torch.stack(query_adjacency_matrix)
            corpus_adjacency_matrix = torch.stack(corpus_adjacency_matrix)
            hinge_score = -torch.sum(
                torch.nn.ReLU()(
                    query_adjacency_matrix - hard_permutation@corpus_adjacency_matrix@hard_permutation.permute(0,2,1)
                ), dim=(1,2)
            )

            all_labels = torch.cat([torch.ones(num_pos_pairs), torch.zeros(num_neg_pairs)])
            positive_histogram.extend(hinge_score[all_labels==1].tolist())
            negative_histogram.extend(hinge_score[all_labels==0].tolist())

    return positive_histogram, negative_histogram

def get_histogram(models_to_run):
    model_name_to_config_map = load_config()

    for model_path, local_config_path in get_models():
        model_params, _ = read_config(local_config_path, with_dict=True)

        model_name = model_params.model
        dataset_name = model_params.dataset[:-6] # TODO: Fix this
        seed = int(model_params.seed)

        if (
            model_name not in models_to_run
        ) or (
            seed != models_to_run[model_name]["seeds"][dataset_name]
        ):
            continue

        if "relevant_models" not in models_to_run[model_name]:
            models_to_run[model_name]["relevant_models"] = []

        models_to_run[model_name]["relevant_models"].append({
            "name": model_name,
            "seed": seed,
            "dataset": dataset_name,
            "model_path": model_path,
            "config_path": model_name_to_config_map[model_name],
        })


    print("Total relevant models found")
    for model_name, metadata in models_to_run.items():
        print(model_name, ":", len(metadata["relevant_models"]) if "relevant_models" in metadata else 0)


    device = 'cuda:1'

    for model_name in models_to_run:
        if "relevant_models" not in models_to_run[model_name]:
            continue
        for idx, relevant_model in enumerate(models_to_run[model_name]["relevant_models"]):

            print(
                "Running Model",
                "\nname:", model_name,
                "\nseed:", relevant_model["seed"],
                "\ndataset:", relevant_model["dataset"],
                "\npath:", relevant_model["model_path"],
                "\nconfig:", relevant_model["config_path"],
            )

            dataset_map = load_datasets(models_to_run[model_name]["data_type"])
            test_dataset = dataset_map[relevant_model["dataset"]]
            model_params, _ = read_config(relevant_model["config_path"], with_dict=True)
            config = model_params.model_config

            model = get_model(
                model_name=model_name,
                config=config,
                max_node_set_size=test_dataset.max_node_set_size,
                max_edge_set_size=test_dataset.max_edge_set_size,
                device=device
            )

            try:
                checkpoint = torch.load(relevant_model["model_path"])
            except:
                print("Could not load model from path:", relevant_model["model_path"])
                continue

            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

            positive_histogram, negative_histogram = eval_node_alignment(
                model=model,
                dataset=test_dataset,
                max_node_set_size=test_dataset.max_node_set_size
            )

            with open(f"positive_histogram_{relevant_model['dataset']}.pkl", "wb") as f:
                pickle.dump(positive_histogram, f)

            with open(f"negative_histogram_{relevant_model['dataset']}.pkl", "wb") as f:
                pickle.dump(negative_histogram, f)

    return models_to_run


def main(table_num):
    
    with open(f"table_{table_num}.json", "rb") as f:
        table_meta = json.load(f)

    get_histogram(table_meta)


if __name__ == "__main__":
    TABLE_NUM = 0
    main(TABLE_NUM)
