import math
import torch
import torch.nn as nn
import torch.nn.functional as F   
from torch_geometric.nn import GINConv
from torch_geometric.data import Batch
from utils import model_utils
from subgraph_matching.models._template import EncodingLayer, InteractionLayer, EncodeThenInteractModel


class EncodingLayerEric(EncodingLayer):
    def __init__(self, input_dim, filters, dropout):
        super(EncodingLayerEric, self).__init__()
        self.input_dim = input_dim
        self.filters = filters
        self.num_filter = len(self.filters)
        self.dropout = dropout

        self.gnn_list = nn.ModuleList()
        self.gnn_filters = [self.input_dim] + self.filters
        for i in range(self.num_filter):
            self.gnn_list.append(
                GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(self.gnn_filters[i], self.gnn_filters[i+1]),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.gnn_filters[i+1], self.gnn_filters[i+1]),
                        torch.nn.BatchNorm1d(self.gnn_filters[i+1]),
                    ),
                    eps=True
                )
            )

        self.mlp_list_inner = nn.ModuleList()
        self.mlp_list_outer = nn.ModuleList()
        for i in range(self.num_filter):
            self.mlp_list_inner.append(nn.Linear(filters[i], filters[i]))
            self.mlp_list_outer.append(nn.Linear(filters[i], filters[i]))

    def regularizer(self, query_node_features_list, corpus_node_features_list, query_graph_features_list, corpus_graph_features_list, query_graph_idx, corpus_graph_idx, batch_size):
        query_node_features = torch.cat(query_node_features_list, dim=1)
        corpus_node_features = torch.cat(corpus_node_features_list, dim=1)
        query_graph_features = torch.cat(query_graph_features_list, dim=1)
        corpus_graph_features = torch.cat(corpus_graph_features_list, dim=1)

        sim_1 = torch.matmul(query_node_features, query_graph_features.t())
        cross_1 = torch.matmul(query_node_features, corpus_graph_features.t())
        sim_2 = torch.matmul(corpus_node_features, corpus_graph_features.t())
        cross_2 = torch.matmul(corpus_node_features, query_graph_features.t())

        mask_1 = torch.zeros_like(sim_1)
        mask_2 = torch.zeros_like(sim_2)
        for node, graph in enumerate(zip(query_graph_idx, corpus_graph_idx)):
            graph_1, graph_2 = graph
            mask_1[node][graph_1] = 1.
            mask_2[node][graph_2] = 1.

        def expectation(sim):
            return (math.log(2.) - F.softplus(-sim)).sum()

        gamma_1 = expectation(sim_1 * mask_1) - expectation(cross_1 * mask_1)
        gamma_2 = expectation(sim_2 * mask_2) - expectation(cross_2 * mask_2)
        return gamma_1 - gamma_2

    def encoder_forward(self, x, edge_index, layer_index):
        x = self.gnn_list[layer_index](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = self.dropout, training=self.training)
        return x

    def deepset_forward(self, x, graph_idx, batch_size, layer_index):
        x = F.relu(self.mlp_list_inner[layer_index](x))
        x = model_utils.unsorted_segment_sum(x, graph_idx, batch_size)
        x = F.relu(self.mlp_list_outer[layer_index](x))
        return x

    def forward(self, graphs, batch_size):
        graph_batch = Batch.from_data_list(graphs)
        features, edge_index, graph_idx = graph_batch.x, graph_batch.edge_index, graph_batch.batch

        layer_wise_node_features = [] # (num_layers, total_num_nodes, num_features)
        layer_wise_graph_features = [] # (num_layers, batch_size, num_features)
        for layer_index in range(self.num_filter):
            features = self.encoder_forward(features, edge_index, layer_index)
            layer_wise_node_features.append(features)
            pooled_features = self.deepset_forward(features, graph_idx, batch_size, layer_index)         
            layer_wise_graph_features.append(pooled_features)
        return layer_wise_graph_features, layer_wise_node_features


class InteractionLayerEric(InteractionLayer):
    class TensorNetworkInteraction(torch.nn.Module):
        def __init__(self, filters, tensor_neurons):
            super(InteractionLayerEric.TensorNetworkInteraction, self).__init__()

            self.last_filter = filters[-1]
            self.tensor_neurons = tensor_neurons

            self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.last_filter, self.last_filter, self.tensor_neurons))
            self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 2 * self.last_filter))
            self.bias = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 1))

            self.score_sim_layer = nn.Sequential(
                                    nn.Linear(tensor_neurons, tensor_neurons),
                                    nn.ReLU(),
                                    nn.Linear(tensor_neurons, 1)
                                )

            torch.nn.init.xavier_uniform_(self.weight_matrix)
            torch.nn.init.xavier_uniform_(self.weight_matrix_block)
            torch.nn.init.xavier_uniform_(self.bias)

        def forward(self, query_features, corpus_features, batch_size):
            scoring = torch.matmul(query_features, self.weight_matrix.view(self.last_filter, -1))
            scoring = scoring.view(batch_size, self.last_filter, -1).permute([0, 2, 1])
            scoring = torch.matmul(scoring, corpus_features.view(batch_size, self.last_filter, 1)).view(batch_size, -1)

            combined_representation = torch.cat((query_features, corpus_features), 1)
            block_scoring = torch.t(torch.mm(self.weight_matrix_block, torch.t(combined_representation)))

            scores = F.relu(scoring + block_scoring + self.bias.view(-1))
            scores = self.score_sim_layer(scores).squeeze()
            return scores


    class MinkowskiInteraction(torch.nn.Module):
        def __init__(self, filters, reduction, dropout, use_conv_stack=True):
            super(InteractionLayerEric.MinkowskiInteraction, self).__init__()

            self.filters = filters
            self.channel_dim = sum(self.filters)
            self.reduction = reduction
            self.dropout = dropout
            self.use_conv_stack = use_conv_stack

            if self.use_conv_stack:
                self.conv_stack = nn.Sequential(
                                    nn.Linear(self.channel_dim, self.channel_dim // self.reduction),
                                    nn.ReLU(),
                                    nn.Dropout(p = self.dropout),
                                    nn.Linear(self.channel_dim // self.reduction, (self.channel_dim // self.reduction) ),
                                    nn.Dropout(p = self.dropout),
                                    nn.Tanh(),
                                )

            self.score_layer = nn.Sequential(
                                    nn.Linear((self.channel_dim // self.reduction) , 16),
                                    nn.ReLU(),
                                    nn.Linear(16 , 1)
                                )

        def forward(self, query_features, corpus_features):
            diff_rep = torch.exp(-torch.pow(query_features - corpus_features, 2))
            if self.use_conv_stack:
                # Not covered in the paper, but found in the official code implementation
                diff_rep = self.conv_stack(diff_rep).squeeze()
            score = self.score_layer(diff_rep).view(-1)
            return score


    def __init__(self, filters, tensor_neurons, reduction, dropout):
        super(InteractionLayerEric, self).__init__()

        self.filters = filters
        self.tensor_neurons = tensor_neurons
        self.reduction = reduction
        self.dropout = dropout

        self.tensor_network_interaction = self.TensorNetworkInteraction(self.filters, self.tensor_neurons)
        self.minkowski_interaction = self.MinkowskiInteraction(self.filters, self.reduction, self.dropout)

        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

        nn.init.zeros_(self.alpha)
        nn.init.zeros_(self.beta)

    def forward(self, query_features, corpus_features, batch_size):
        score_1 = self.tensor_network_interaction(query_features[-1], corpus_features[-1], batch_size)
        score_2 = self.minkowski_interaction(
                    torch.cat(query_features, dim=1),
                    torch.cat(corpus_features, dim=1)
                )
        score = self.alpha * score_1 + self.beta * score_2
        return score


class ERIC(EncodeThenInteractModel):
    def __init__(
        self,
        input_dim,
        max_node_set_size,
        max_edge_set_size,
        gnn_filters,
        reduction,
        tensor_neurons,
        dropout,
        device
    ):
        super(ERIC, self).__init__()

        self.encoding_layer = EncodingLayerEric(input_dim, gnn_filters, dropout)
        self.interaction_layer = InteractionLayerEric(gnn_filters, tensor_neurons, reduction, dropout)
        self.gamma = nn.Parameter(torch.Tensor(1)) 
        nn.init.zeros_(self.gamma)

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        batch_size = len(graph_sizes)
        query_graphs, corpus_graphs = zip(*graphs)

        # Encoding graph level features
        query_graph_features, query_node_features = self.encoding_layer(query_graphs, batch_size)
        corpus_graph_features, corpus_node_features = self.encoding_layer(corpus_graphs, batch_size)

        # Interaction
        score = self.interaction_layer(query_graph_features, corpus_graph_features, batch_size)

        # Regularizer Term
        query_graph_idx = Batch.from_data_list(query_graphs).batch
        corpus_graph_idx = Batch.from_data_list(corpus_graphs).batch
        self.regularizer = self.gamma * self.encoding_layer.regularizer(query_node_features, corpus_node_features, query_graph_features, corpus_graph_features, query_graph_idx, corpus_graph_idx, batch_size)
        return score
