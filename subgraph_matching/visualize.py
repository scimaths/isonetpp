import os
import time
import json
import torch
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils.tooling import seed_everything
from subgraph_matching.test import evaluate_model
from subgraph_matching.dataset import get_datasets
from subgraph_matching.model_handler import get_model, get_data_type_for_model

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from matplotlib.colors import LinearSegmentedColormap
cmap_custom = LinearSegmentedColormap.from_list('custom', ['red', 'black', 'green'], N=256)

vis_dir_name = "visualizations/"

def visualize_model(model, dataset, device='cuda', num_vis=10):
    model.to(device)
    model.eval()

    pos_pairs, neg_pairs = dataset.pos_pairs, dataset.neg_pairs
    all_pairs = pos_pairs + neg_pairs
    num_pos_pairs, num_neg_pairs = len(pos_pairs), len(neg_pairs)

    all_adj_matrices = []
    predictions = []
    all_transport_plans = []
    all_graph_sizes = []
    num_batches = dataset.create_custom_batches(all_pairs)
    for batch_idx in range(num_batches):
        batch_graphs, batch_graph_sizes, _, batch_adj_matrices = dataset.fetch_batch_by_id(batch_idx)
        pred, transport_plan = model.forward_with_alignment(batch_graphs, batch_graph_sizes, batch_adj_matrices)
        all_adj_matrices.extend(batch_adj_matrices)
        all_graph_sizes.extend(batch_graph_sizes)
        predictions.append(pred.data)
        all_transport_plans.append(transport_plan)
    all_predictions = torch.cat(predictions, dim=0)
    all_labels = torch.cat([torch.ones(num_pos_pairs), torch.zeros(num_neg_pairs)])
    q_to_c_plans = torch.cat(list(map(lambda x: x[0], all_transport_plans)), dim=0)
    c_to_q_plans = torch.cat(list(map(lambda x: x[1], all_transport_plans)), dim=0)

    chosen_indices = np.random.permutation(np.arange(len(all_predictions), dtype=int))[:num_vis]
    for idx in chosen_indices:
        pred = all_predictions[idx]
        label = all_labels[idx]
        adj_matrices = all_adj_matrices[idx]
        transport_plans = q_to_c_plans[idx].detach().cpu().numpy(), c_to_q_plans[idx].detach().cpu().numpy()
        query_size, corpus_size = all_graph_sizes[idx]

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))

        query_graph = nx.from_numpy_array(adj_matrices[0].detach().cpu().numpy()[:query_size, :query_size])
        nx.draw(query_graph, nx.spring_layout(query_graph), with_labels=True, labels={node: str(node) for node in query_graph}, node_size=1000, font_size=25, font_color='w', ax=axs[0, 0])
        axs[0, 0].set_title('Query graph')

        corpus_graph = nx.from_numpy_array(adj_matrices[1].detach().cpu().numpy()[:corpus_size, :corpus_size])
        nx.draw(corpus_graph, nx.spring_layout(corpus_graph), with_labels=True, labels={node: str(node) for node in corpus_graph}, node_size=1000, font_size=25, font_color='w', ax=axs[0, 1])
        axs[0, 1].set_title('Corpus graph')

        im1 = axs[1, 0].imshow(transport_plans[0], cmap=cmap_custom, interpolation='none', aspect='auto')
        axs[1, 0].set_title('Q -> C matrix')

        im2 = axs[1, 1].imshow(transport_plans[1], cmap=cmap_custom, interpolation='none', aspect='auto')
        axs[1, 1].set_title('C -> Q matrix')

        for ax_idx, ax in enumerate(axs[1]):
            labels = np.arange(transport_plans[0].shape[1])
            ax.set_xticks(labels)
            ax.set_yticks(labels)
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            for i in range(len(labels)):
                for j in range(len(labels)):
                    text = ax.text(j, i, f'{transport_plans[ax_idx][i, j]:.2f}', ha='center', va='center', color='w')

        plt.tight_layout()
        fig.suptitle(f"Index - {idx}, Label - {label}, Score - {round(all_predictions[idx].item(), 6)}", fontsize=25)
        plt.savefig(os.path.join(vis_dir_name, f"{idx}.png"))
        plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_loc")
    parser.add_argument("--config_path")
    parser.add_argument("--dataset", default='aids')
    parser.add_argument("--num_vis", type=int, default=100)
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    device = 'cuda'
    
    dataset_config = dict(
        dataset_name = args.dataset,
        dataset_size = "large",
        dataset_base_path = ".",
        batch_size = 128,
    )
    saved_config_path = args.model_loc.replace(".pth", ".json").replace("trained_models", "configs")
    model_name = json.load(open(saved_config_path))['model']
    vis_dir_name = os.path.join(vis_dir_name, model_name)
    os.makedirs(vis_dir_name, exist_ok=True)
    data_type = get_data_type_for_model(model_name)
    datasets = get_datasets(dataset_config, None, data_type)

    model = get_model(
        model_name=model_name,
        config_path=args.config_path,
        max_node_set_size=datasets['train'].max_node_set_size,
        max_edge_set_size=datasets['train'].max_edge_set_size,
        device=device
    )
    seed_everything(args.seed)
    model.load_state_dict(torch.load(args.model_loc)['model_state_dict'])
    visualize_model(model, datasets['val'], device, num_vis=args.num_vis)