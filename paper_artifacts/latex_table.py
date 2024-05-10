import os
import json
import torch
import pickle
import numpy as np
import networkx as nx
import seaborn as sns
from utils import model_utils
import matplotlib.pyplot as plt
from utils.tooling import read_config
from utils.tooling import seed_everything
import networkx.algorithms.isomorphism as iso
from utils.eval import compute_average_precision
from subgraph_matching.model_handler import get_model
from subgraph_matching.dataset import SubgraphIsomorphismDataset


def get_models(
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
    standard_deviation = np.std(per_query_hits_at_k)
    standard_error = standard_deviation / np.sqrt(len(per_query_hits_at_k))
    return mean_hits_at_k, standard_error


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
        for dataset_name in ["aids", "mutag", "ptc_fm", "ptc_fr", "ptc_mm", "ptc_mr"]:
            dataset_std_error_scores = []
            for relevant_model in relevant_models:
                if relevant_model["dataset"] == dataset_name:
                    dataset_std_error_scores.append(relevant_model["std_error"])
            std_errors = [str(round(float(x), 3)) for x in dataset_std_error_scores]
            print(" | ".join(std_errors), end="")
            if dataset_name != "ptc_mr":
                print(" & ", end="")
        print("\\\\")
    print(table_end)


def evaluate_improvement_nodes(model, dataset, dataset_name):
    def get_alignment_nodes(mapping, num_query_nodes, num_corpus_nodes, device):

        p_hat = torch.zeros((num_query_nodes, num_corpus_nodes), device=device)

        for key in mapping.keys():
            p_hat[mapping[key]][key] = 1

        return p_hat

    def get_norm_qc_pair_nodes(query_edges, corpus_edges, transport_plans, num_query_nodes, num_corpus_nodes):

        Query = nx.Graph()
        Query.add_edges_from(query_edges)

        Corpus = nx.Graph()
        Corpus.add_edges_from(corpus_edges)

        GM = iso.GraphMatcher(Corpus,Query)

        best_p_hat = None
        best_norm_final_transport = torch.zeros(1)
        for mapping in GM.subgraph_isomorphisms_iter():
            p_hat = get_alignment_nodes(mapping, num_query_nodes, num_corpus_nodes, transport_plans.device)
            norm = torch.sum(transport_plans[-1, :num_query_nodes, :num_corpus_nodes] * p_hat)
            if norm >= best_norm_final_transport:
                best_norm_final_transport = norm
                best_p_hat = p_hat

        assert best_p_hat is not None
        norm = torch.sum(transport_plans[:, :num_query_nodes, :num_corpus_nodes] * best_p_hat.unsqueeze(0), dim=(1,2))
        return norm.tolist()

    model.eval()

    pos_pairs, neg_pairs = dataset.pos_pairs, dataset.neg_pairs

    num_query_graphs = len(dataset.query_graphs)

    norms_total = []
    for query_idx in range(num_query_graphs):
        pos_pairs_for_query = list(filter(lambda pair: pair[0] == query_idx, pos_pairs))

        if len(pos_pairs_for_query) > 0:

            num_batches = dataset.create_custom_batches(pos_pairs_for_query)

            norms_per_query = []
            for batch_idx in range(num_batches):
                batch_graphs, batch_graph_sizes, _, batch_adj_matrices = dataset.fetch_batch_by_id(batch_idx)
                _, _, from_idx, to_idx, graph_idx = model_utils.get_graph_features(batch_graphs)

                batch_data_sizes_flat = [item for sublist in batch_graph_sizes for item in sublist]

                edge_counts  = model_utils.get_paired_edge_counts(from_idx, to_idx, graph_idx, 2*len(batch_graph_sizes))
                edge_counts = [item for sublist in edge_counts for item in sublist]

                from_idx_suff = torch.cat([
                    torch.tensor([sum(batch_data_sizes_flat[:node_idx])]).repeat(edge_counts[node_idx])
                    for node_idx in range(len(edge_counts))
                ])

                from_idx = from_idx.cpu() -  from_idx_suff
                to_idx = to_idx.cpu() - from_idx_suff

                transport_plans = model.forward_for_alignment(batch_graphs, batch_graph_sizes, batch_adj_matrices).cpu()

                batch_size = transport_plans.shape[0]
                for batch_idx in range(batch_size):
                    query_from = from_idx[sum(edge_counts[:2*batch_idx]):sum(edge_counts[:2*batch_idx+1])].tolist()
                    query_to = to_idx[sum(edge_counts[:2*batch_idx]):sum(edge_counts[:2*batch_idx+1])].tolist()

                    corpus_from = from_idx[sum(edge_counts[:2*batch_idx+1]):sum(edge_counts[:2*batch_idx+2])].tolist()
                    corpus_to = to_idx[sum(edge_counts[:2*batch_idx+1]):sum(edge_counts[:2*batch_idx+2])].tolist()

                    query_edges = [(query_from[idx], query_to[idx]) for idx in range(len(query_from))]
                    corpus_edges = [(corpus_from[idx], corpus_to[idx]) for idx in range(len(corpus_from))]

                    norms_per_pair = get_norm_qc_pair_nodes(query_edges, corpus_edges, transport_plans[batch_idx], batch_data_sizes_flat[batch_idx*2], batch_data_sizes_flat[batch_idx*2+1])
                    norms_per_query.append(norms_per_pair)

            norms_total.extend(norms_per_query)
    values_np = np.array(norms_total)

    if not os.path.exists('histogram_dumps_node_early_baseline'):
        os.makedirs('histogram_dumps_node_early_baseline')
        os.makedirs('histogram_plots_node_early_baseline')

    for time in range(values_np.shape[1]):
        pickle.dump(values_np[:,time], open(f'histogram_dumps_node_early_baseline/model_time_{dataset_name}_{time}', 'wb'))
        sns.histplot(values_np[:,time], binwidth=0.1, binrange=(0, np.ceil(np.max(values_np))), kde=True, label=f'time = {time}', palette='pastel')

    # Adding labels and title
    plt.xlabel('EntryWise_L1Norm(P * P_hat)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram: Node Early ({dataset_name})')

    # Show legend
    plt.legend()
    plt.grid(True)
    plt.savefig(f'histogram_plots_node_early_baseline/histogram_{dataset_name}.png')
    plt.clf()

def evaluate_model(model, dataset):
    model.eval()

    # Compute global statistics
    pos_pairs, neg_pairs = dataset.pos_pairs, dataset.neg_pairs
    average_precision = compute_average_precision(model, pos_pairs, neg_pairs, dataset)

    # Compute per-query statistics
    num_query_graphs = len(dataset.query_graphs)
    per_query_avg_prec = []

    for query_idx in range(num_query_graphs):
        pos_pairs_for_query = list(filter(lambda pair: pair[0] == query_idx, pos_pairs))
        neg_pairs_for_query = list(filter(lambda pair: pair[0] == query_idx, neg_pairs))

        if len(pos_pairs_for_query) > 0 and len(neg_pairs_for_query) > 0:
            per_query_avg_prec.append(
                compute_average_precision(model, pos_pairs_for_query, neg_pairs_for_query, dataset)
            )
    mean_average_precision = np.mean(per_query_avg_prec)

    standard_deviation = np.std(per_query_avg_prec)
    standard_error = standard_deviation / np.sqrt(len(per_query_avg_prec))
    return average_precision, mean_average_precision, standard_error

def evaluate_improvement_edges(model, dataset, dataset_name):

    def get_alignment_edges(mapping, query_edges, corpus_edges, device):

        num_query_edges = len(query_edges)
        num_corpus_edges = len(corpus_edges)

        edges_corpus_to_idx = {corpus_edges[idx]: idx for idx in range(len(corpus_edges))}
        reverse_mapping = {mapping[key]: key for key in mapping}

        s_hat = torch.zeros((num_query_edges, num_corpus_edges), device=device)

        for edge_idx in range(num_query_edges):

            corpus_edge_1 = (reverse_mapping[query_edges[edge_idx][0]], reverse_mapping[query_edges[edge_idx][1]])
            corpus_edge_2 = (reverse_mapping[query_edges[edge_idx][1]], reverse_mapping[query_edges[edge_idx][0]])

            if corpus_edge_1 in edges_corpus_to_idx:
                s_hat[edge_idx][edges_corpus_to_idx[corpus_edge_1]] = 1

            if corpus_edge_2 in edges_corpus_to_idx:
                s_hat[edge_idx][edges_corpus_to_idx[corpus_edge_2]] = 1

        return s_hat

    def get_norm_qc_pair_edges(query_edges, corpus_edges, transport_plans):

        Query = nx.Graph()
        Query.add_edges_from(query_edges)

        Corpus = nx.Graph()
        Corpus.add_edges_from(corpus_edges)

        GM = iso.GraphMatcher(Corpus,Query)

        best_s_hat = None
        best_norm_final_transport = torch.zeros(1)
        for mapping in GM.subgraph_isomorphisms_iter():
            p_hat = get_alignment_edges(mapping, query_edges, corpus_edges, transport_plans.device)
            norm = torch.sum(transport_plans[-1, :len(query_edges), :len(corpus_edges)] * p_hat)
            if norm >= best_norm_final_transport:
                best_norm_final_transport = norm
                best_s_hat = p_hat

        assert best_s_hat is not None
        norm = torch.sum(transport_plans[:, :len(query_edges), :len(corpus_edges)] * best_s_hat.unsqueeze(0), dim=(1,2))
        return norm.tolist()

    model.eval()

    pos_pairs, neg_pairs = dataset.pos_pairs, dataset.neg_pairs

    num_query_graphs = len(dataset.query_graphs)

    norms_total = []
    for query_idx in range(num_query_graphs):
        pos_pairs_for_query = list(filter(lambda pair: pair[0] == query_idx, pos_pairs))

        if len(pos_pairs_for_query) > 0:

            num_batches = dataset.create_custom_batches(pos_pairs_for_query)

            norms_per_query = []
            for batch_idx in range(num_batches):
                batch_graphs, batch_graph_sizes, _, batch_adj_matrices = dataset.fetch_batch_by_id(batch_idx)
                _, _, from_idx, to_idx, graph_idx = model_utils.get_graph_features(batch_graphs)


                batch_data_sizes_flat = [item for sublist in batch_graph_sizes for item in sublist]

                edge_counts  = model_utils.get_paired_edge_counts(from_idx, to_idx, graph_idx, 2*len(batch_graph_sizes))
                edge_counts = [item for sublist in edge_counts for item in sublist]

                from_idx_suff = torch.cat([
                    torch.tensor([sum(batch_data_sizes_flat[:node_idx])]).repeat(edge_counts[node_idx])
                    for node_idx in range(len(edge_counts))
                ])

                from_idx = from_idx.cpu() - from_idx_suff
                to_idx = to_idx.cpu() - from_idx_suff

                transport_plans = model(batch_graphs, batch_graph_sizes, batch_adj_matrices).cpu()
                batch_size = transport_plans.shape[0]

                transport_plans = model.forward_for_alignment(batch_graphs, batch_graph_sizes, batch_adj_matrices).cpu()

                batch_size = transport_plans.shape[0]
                for batch_idx in range(batch_size):
                    query_from = from_idx[sum(edge_counts[:2*batch_idx]):sum(edge_counts[:2*batch_idx+1])].tolist()
                    query_to = to_idx[sum(edge_counts[:2*batch_idx]):sum(edge_counts[:2*batch_idx+1])].tolist()

                    corpus_from = from_idx[sum(edge_counts[:2*batch_idx+1]):sum(edge_counts[:2*batch_idx+2])].tolist()
                    corpus_to = to_idx[sum(edge_counts[:2*batch_idx+1]):sum(edge_counts[:2*batch_idx+2])].tolist()

                    query_edges = [(query_from[idx], query_to[idx]) for idx in range(len(query_from))]
                    corpus_edges = [(corpus_from[idx], corpus_to[idx]) for idx in range(len(corpus_from))]

                    norms_per_pair = get_norm_qc_pair_edges(query_edges, corpus_edges, transport_plans[batch_idx])
                    norms_per_query.append(norms_per_pair)

            norms_total.extend(norms_per_query)
    values_np = np.array(norms_total)

    if not os.path.exists('histogram_dumps_edge_early_baseline'):
        os.makedirs('histogram_dumps_edge_early_baseline')
        os.makedirs('histogram_plots_edge_early_baseline')

    for time in range(values_np.shape[1]):
        pickle.dump(values_np[:,time], open(f'histogram_dumps_edge_early_baseline/model_time_{dataset_name}_{time}', 'wb'))
        sns.histplot(values_np[:,time], binwidth=0.1, binrange=(0, np.ceil(np.max(values_np))), kde=True, label=f'time = {time}', palette='pastel')

    # Adding labels and title
    plt.xlabel('EntryWise_L1Norm(S * S_hat)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram: Edge Early ({dataset_name})')

    # Show legend
    plt.legend()
    plt.grid(True)
    plt.savefig(f'histogram_dumps_edge_early_baseline/histogram_{dataset_name}.png')
    plt.clf()
    
    return norms_total


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
            "hits@20": "0",
            "std_error": "0"
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

            # Apply the mapping to the loaded state dictionary keys
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)

            seed_everything(relevant_model["seed"])

            # evaluate_improvement_nodes(model, test_dataset, relevant_model["dataset"])

            # evaluate_improvement_edges(model, test_dataset, relevant_model["dataset"])

            _, test_map_score, test_std_error = evaluate_model(model, test_dataset)
            print("Test MAP Score:", test_map_score)
            print("Test Standard Error:", test_std_error)
            models_to_run[model_name]["relevant_models"][idx]["map_score"] = str(test_map_score)
            models_to_run[model_name]["relevant_models"][idx]["std_error"] = str(test_std_error)

            hits_at_20, test_std_error = hits_at_k(model, test_dataset, 20)
            print("Test HITS@20 Score:", hits_at_20, "\n")
            print("Test Standard Error:", test_std_error)
            models_to_run[model_name]["relevant_models"][idx]["hits@20"] = str(hits_at_20)
            models_to_run[model_name]["relevant_models"][idx]["std_error_hits@20"] = str(test_std_error)

    return models_to_run



def main():

    with open(table_path, "rb") as f:
        table_meta = json.load(f)

    table_meta_with_scores = get_scores(table_meta)

    with open(base_path + f"paper_artifacts/table_metadata/table_{table_num}_with_scores.json", "w") as f:
        json.dump(table_meta_with_scores, f)

    dump_latex(table_meta_with_scores)

if __name__ == "__main__":
    # base_path = "/raid/infolab/ashwinr/isonetpp/"
    base_path = "/mnt/home/ashwinr/btp24/grph/gitlab_repo/isonetpp/"
    paths_to_experiment_dir = [
        base_path + "paper_artifacts/collection/"
    ]

    table_num = 4
    table_path = base_path + f"paper_artifacts/table_metadata/table_{table_num}.json"

    collection_path = base_path + "paper_artifacts/collection/"
    main()
