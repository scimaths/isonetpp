import os

import torch
import numpy as np
import pickle
import json
from subgraph_matching import dataset
from subgraph_matching.dataset import SubgraphIsomorphismDataset
from subgraph_matching.test import evaluate_model
from subgraph_matching.model_handler import get_model
from utils.tooling import read_config


def get_models(
    paths_to_experiment_dir = [
        "/mnt/nas/vaibhavraj/isonetpp_experiments/",
        "/mnt/nas/vaibhavraj/isonetpp_experiments_march_16/",
        "/mnt/nas/vaibhavraj/isonet_experiments_02_april/",
        "/mnt/home/vaibhavraj/isonetpp_enhanced_code/experiments/",
        "/mnt/home/vaibhavraj/isonetpp_enhanced_code/experiments_archived_march_16/",
        "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/experiments/",
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


def hits_at_k(model, dataset, k):
    model.eval()

    pos_pairs, neg_pairs = dataset.pos_pairs, dataset.neg_pairs

    num_query_graphs = len(dataset.query_graphs)
    per_query_hits_at_k = []

    for query_idx in range(num_query_graphs):
        pos_pairs_for_query = list(filter(lambda pair: pair[0] == query_idx, pos_pairs))
        neg_pairs_for_query = list(filter(lambda pair: pair[0] == query_idx, neg_pairs))

        if len(pos_pairs_for_query) > 0 and len(neg_pairs_for_query) > 0:
            all_pairs = pos_pairs_for_query + neg_pairs_for_query
            num_pos_pairs, num_neg_pairs = len(pos_pairs_for_query), len(neg_pairs_for_query)

            predictions = []
            num_batches = dataset.create_custom_batches(all_pairs)
            for batch_idx in range(num_batches):
                batch_graphs, batch_graph_sizes, _, batch_adj_matrices = dataset.fetch_batch_by_id(batch_idx)
                predictions.append(model(batch_graphs, batch_graph_sizes, batch_adj_matrices).data)

            all_predictions = torch.cat(predictions, dim=0)
            all_labels = torch.cat([torch.ones(num_pos_pairs), torch.zeros(num_neg_pairs)])

            ranking = torch.argsort(-all_predictions.cpu()).tolist()
            all_labels_ranked = all_labels.cpu()[ranking]

            neg_idx = torch.where(all_labels_ranked == 0)[0]
            # TODO: Fix this
            if len(neg_idx) < k:
                raise ValueError("Not enough negative samples to compute HITS@K")

            k_neg_idx = neg_idx[k-1]
            hits_20 = torch.sum(all_labels_ranked[:k_neg_idx]) / (torch.sum(all_labels_ranked))
            per_query_hits_at_k.append(hits_20)

    mean_hits_at_k = np.mean(per_query_hits_at_k)

    return mean_hits_at_k


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


def dump_latex(table_meta):
    description_len = None
    for model in table_meta:
        description_len = len(table_meta[model]["description"])
        break

    for model in table_meta:
        assert len(table_meta[model]["description"]) == description_len

    table_begin = (
        r"""
        \begin{table}[htbp]
        \centering
        \begin{adjustbox}{width=1\textwidth}
        \begin{tabular}{@{}""" + ("c") * description_len + r"""|cccccc|cccccc@{}}
        \toprule
        \multicolumn{""" + str(description_len) + r"""}{c|}{Model Description} & \multicolumn{6}{c|}{Mean Average Precision (MAP)} & \multicolumn{6}{c}{HITS @ 20} \\ """
        + " ".join(["&"] * (description_len - 1)) + r""" & AIDS & MUTAG & PTC\_FM & PTC\_FR & PTC\_MM & PTC\_MR & AIDS & MUTAG & PTC\_FM & PTC\_FR & PTC\_MM & PTC\_MR \\
        \midrule \midrule
        """
    )

    table_end = (
        r"""
        \bottomrule
        \end{tabular}
        \end{adjustbox}
        \end{table} 
        """
    )
    print(table_begin)
    for model_name in table_meta:
        model = table_meta[model_name]
        print(" & ".join(model["description"]), end=" & ")
        if "relevant_models" not in model:
            print(" & " * 11, end="\\\\\n")
            continue
        relevant_models = model["relevant_models"]
        for dataset_name in ["aids", "mutag", "ptc_fm", "ptc_fr", "ptc_mm", "ptc_mr"]:
            dataset_map_scores = []
            for relevant_model in relevant_models:
                if relevant_model["dataset"] == dataset_name:
                    dataset_map_scores.append(relevant_model["map_score"])
            maps = [str(round(float(x), 3)) for x in dataset_map_scores]
            print(" | ".join(maps), end=" & ")
        for dataset_name in ["aids", "mutag", "ptc_fm", "ptc_fr", "ptc_mm", "ptc_mr"]:
            dataset_hit_scores = []
            for relevant_model in relevant_models:
                if relevant_model["dataset"] == dataset_name:
                    dataset_hit_scores.append(relevant_model["hits@20"])
            hits = [str(round(float(x), 3)) for x in dataset_hit_scores]
            print(" | ".join(hits), end="")
            if dataset_name != "ptc_mr":
                print(" & ", end="")
        print("\\\\")
    print(table_end)


def get_scores(models_to_run):
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
            "map_score": "0",
            "hits@20": "0"
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

            _, test_map_score = evaluate_model(model, test_dataset)
            print("Test MAP Score:", test_map_score)
            models_to_run[model_name]["relevant_models"][idx]["map_score"] = str(test_map_score)

            hits_at_20 = hits_at_k(model, test_dataset, 20)
            print("Test HITS@20 Score:", hits_at_20, "\n")
            models_to_run[model_name]["relevant_models"][idx]["hits@20"] = str(hits_at_20)

    return models_to_run



def main(table_num):
    
    with open(f"table_{table_num}_main.json", "rb") as f:
        table_meta = json.load(f)

    table_meta_with_scores = get_scores(table_meta)

    with open(f"table_{table_num}_with_scores.json", "w") as f:
        json.dump(table_meta_with_scores, f)

    dump_latex(table_meta_with_scores)

if __name__ == "__main__":
    TABLE_NUM = 1
    main(TABLE_NUM)