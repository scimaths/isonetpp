import os

import torch
import numpy as np
import pickle

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
        "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/experiments/"
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
                model_name_to_config_map[model_params.name] = model_params.model_config
    return model_name_to_config_map


def load_datasets():
    dataset_map = {}
    for dataset_name in ["aids", "mutag", "ptc_fm", "ptc_fr", "ptc_mm", "ptc_mr"]:
        dataset_map[dataset_name] = SubgraphIsomorphismDataset(
            mode = 'test',
            dataset_name = dataset_name,
            dataset_size = "large",
            batch_size = 128,
            data_type = dataset.GMN_DATA_TYPE,
            dataset_base_path = ".",
            experiment = None
        )
    return dataset_map


def get_scores(models_to_run, seeds_to_run):
    index = 0
    for model_path, config_path in get_models():
        model_params, _ = read_config(config_path, with_dict=True)
        model_name = model_params.model
        # TODO: Fix this
        dataset_name = model_params.dataset[:-6]
        seed = int(model_params.seed)

        if model_name not in models_to_run or seed != seeds_to_run[dataset_name]:
            continue
        else:
            index += 1

    print("Total relevant models found", index)


    model_name_to_config_map = load_config()
    dataset_map = load_datasets()
    device = 'cuda'

    map_scores = {}
    hits_at_20_scores = {}
    corresp_models = {}
    
    for model_path, config_path in get_models():

        model_params, _ = read_config(config_path, with_dict=True)
        model_name = model_params.model
        # TODO: Fix this
        dataset_name = model_params.dataset[:-6]
        seed = int(model_params.seed)

        if model_name not in models_to_run or seed != seeds_to_run[dataset_name]:
            continue

        config = model_name_to_config_map[model_name]

        print(
            "Running Model",
            "\nname:", model_name,
            "\nseed:", seed,
            "\ndataset:", dataset_name,
            "\npath:", model_path,
            "\nconfig reference:", config_path,
            # "\nconfig:", config.toJSON()
        )


        test_dataset = dataset_map[dataset_name]

        model = get_model(
            model_name=model_name,
            config=config,
            max_node_set_size=test_dataset.max_node_set_size,
            max_edge_set_size=test_dataset.max_edge_set_size,
            device=device
        )

        try:
            checkpoint = torch.load(model_path)
        except:
            print("Could not load model from path:", model_path)
            continue

        
        model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)

        _, test_map_score = evaluate_model(model, test_dataset)
        print("Test MAP Score:", test_map_score)

        hits_at_20 = hits_at_k(model, test_dataset, 20)
        print("Test HITS@20 Score:", hits_at_20, "\n")


        if model_name not in map_scores:
            map_scores[model_name] = {
                "aids": [],
                "mutag": [],
                "ptc_fm": [],
                "ptc_fr": [],
                "ptc_mm": [],
                "ptc_mr": []
            }
            hits_at_20_scores[model_name] = {
                "aids": [],
                "mutag": [],
                "ptc_fm": [],
                "ptc_fr": [],
                "ptc_mm": [],
                "ptc_mr": []
            }
            corresp_models[model_name] = {
                "aids": [],
                "mutag": [],
                "ptc_fm": [],
                "ptc_fr": [],
                "ptc_mm": [],
                "ptc_mr": []
            }

        map_scores[model_name][dataset_name].append(test_map_score)
        hits_at_20_scores[model_name][dataset_name].append(hits_at_20)
        corresp_models[model_name][dataset_name].append(model_path)

    return map_scores, hits_at_20_scores, corresp_models


if __name__ == "__main__":
    table_3_models = [
        'gmn_baseline_scoring=agg___tp=attention_pp=identity_when=post',
        'gmn_baseline_scoring=agg___tp=attention_pp=identity_when=pre',
        'gmn_baseline_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=post___unify=true',
        'gmn_baseline_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true'
        'gmn_iterative_refinement_scoring=agg___tp=attention_pp=identity_when=post',
        'gmn_iterative_refinement_scoring=agg___tp=attention_pp=identity_when=pre',
        'gmn_iterative_refinement_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=post___unify=true',
        'gmn_iterative_refinement_scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true',
        'edge_early_interaction'
    ]

    table_3_seeds = {
        "aids": 7762,
        "mutag": 7762,
        "ptc_fm": 7474,
        "ptc_fr": 7762,
        "ptc_mm": 7762,
        "ptc_mr": 7366
    }

    map_scores, hits_at_20, corresp_models = get_scores(table_3_models, table_3_seeds)

    with open("table_3_scores.pkl", "wb") as f:
        pickle.dump({
            "map_scores": map_scores,
            "hits_at_20": hits_at_20,
            "corresp_models": corresp_models
        }, f)



    TABLE_3_LATEX_BEGIN = """
    \begin{table}[htbp]
        \centering
        \begin{adjustbox}{width=1\textwidth}
            \begin{tabular}{@{}*{1}l|*{6}llllll|*{6}llllll@{}}
                \toprule
                &
                \multicolumn{6}{c}{Mean Average Precision (MAP)} &
                \multicolumn{6}{c}{HITS @ 10} \\ 
                &
                AIDS &
                MUTAG &
                PTC\_FM &
                PTC\_FR &
                PTC\_MM &
                PTC\_MR &
                AIDS &
                MUTAG &
                PTC\_FM &
                PTC\_FR &
                PTC\_MM &
                PTC\_MR \\
                \midrule \midrule
    """

    TABLE_3_LATEX_END = """
            \bottomrule
            \end{tabular}
        \end{adjustbox}
    \end{table} 
    """

    for model in table_3_models:
        if model in map_scores:
            print(model)
            for dataset_name in ["aids", "mutag", "ptc_fm", "ptc_fr", "ptc_mm", "ptc_mr"]:
                maps = map_scores[model][dataset_name]
                maps = [str(round(x, 3)) for x in maps]
                print(" | ".join(maps), end=" ")
                print(" & ", end="")
            for dataset_name in ["aids", "mutag", "ptc_fm", "ptc_fr", "ptc_mm", "ptc_mr"]:
                hits = hits_at_20[model][dataset_name]
                hits = [str(round(x, 3)) for x in hits]
                print(" | ".join(hits), end=" ")
                if dataset_name != "ptc_mr":
                    print(" & ", end="")
            print("\\\\")
