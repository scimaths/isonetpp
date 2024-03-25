import torch
import torch.nn as nn
import torch.nn.functional as F   
from torch_geometric.nn import GINConv
from torch_geometric.data import Batch


class EncodingLayer:
    def __init__(self, input_dim, filters):
        super(EncodingLayer, self).__init__()
        self.num_filter = len(filters)

        self.gnn_list = nn.ModuleList()
        gnn_filters = [input_dim] + filters
        for i in range(self.num_filter):
            self.gnn_list.append(
                GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(gnn_filters[i], gnn_filters[i+1]),
                        torch.nn.ReLU(),
                        torch.nn.Linear(gnn_filters[i+1], gnn_filters[i+1]),
                        torch.nn.BatchNorm1d(gnn_filters[i+1]),
                    ),eps=True
                )
            )

        self.mlp_list_inner = nn.ModuleList()
        self.mlp_list_outer = nn.ModuleList()
        for i in range(self.num_filter):
            self.mlp_list_inner.append(nn.Linear(filters[i], filters[i]))
            self.mlp_list_outer.append(nn.Linear(filters[i], filters[i]))

    def regularizer(self, query_node_features, corpus_node_features, query_graph_features, corpus_graph_features, query_graph_idx, corpus_graph_idx, batch_size):
        query_node_features = torch.stack(query_node_features, dim=0)
        corpus_node_features = torch.stack(corpus_node_features, dim=0)
        query_graph_features = torch.stack(query_graph_features, dim=0)
        corpus_graph_features = torch.stack(corpus_graph_features, dim=0)

        gamma_i = torch.abs(
                    torch.sub(
                        torch.bmm(query_node_features, query_graph_features.t()), # (num_layers, total_num_nodes, batch_size)
                        torch.bmm(query_node_features, corpus_graph_features.t()), # (num_layers, total_num_nodes, batch_size)
                    )
                )
        gamma_i = model_utils.unsorted_segment_sum(gamma_i, query_graph_idx, batch_size) # (num_layers, batch_size, batch_size)

        gamma_j = torch.sub(
            torch.bmm(corpus_node_features, query_graph_features.t()), # (num_layers, total_num_nodes, batch_size)
            torch.bmm(corpus_node_features, corpus_graph_features.t()), # (num_layers, total_num_nodes, batch_size)
        )
        gamma_j = model_utils.unsorted_segment_sum(gamma_j, corpus_graph_idx, batch_size) # (num_layers, batch_size, batch_size)

        gamma = torch.sum(
                    gamma_i,
                    gamma_j,
                    torch.abs(gamma_i - gamma_j),
                ) / self.num_filter

        return gamma

    def encoder_forward(self, x, edge_index, layer_index):
        x = self.gnn_list[i](x, edge_index)
        x = F.relu(layer_index)
        x = F.dropout(x, p = self.dropout, training=self.training)
        return x

    def deepset_forward(self, x, graph_idx, batch_size, layer_index):
        x = F.relu(self.mlp_list_inner[layer_index](x))
        x = model_utils.unsorted_segment_sum(x, graph_idx, batch_size)
        x = F.relu(self.mlp_list_outer[layer_index](x))
        return x

    def forward(self, x, edge_index, graph_idx, batch_size):
        layer_wise_node_features = [] # (num_layers, total_num_nodes, num_features)
        layer_wise_graph_features = [] # (num_layers, batch_size, num_features)
        for layer_index in range(self.num_filter):
            x = self.encoder_forward(x, edge_index, layer_index)
            layer_wise_node_features.append(x)
            x = self.deepset_forward(x, graph_idx, batch_size, layer_index)         
            layer_wise_graph_features.append(x)
        return layer_wise_graph_features, layer_wise_node_features


class InteractionLayer:
    class TensorNetworkInteraction:
        def __init__(self, filters, tensor_neurons):
            super(TensorNetworkInteraction, self).__init__()

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


    class MinkowskiInteraction:
        def __init__(self, filters, reduction, dropout, use_conv_stack=True):
            super(MinkowskiInteraction, self).__init__()

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


    def __init__(self, filters, tensor_neurons, reduction):
        super(InteractionLayer, self).__init__()

        self.filters = filters
        self.tensor_neurons = tensor_neurons
        self.reduction = reduction

        self.TensorNetworkInteraction = TensorNetworkInteraction(self.tensor_neurons, self.filters)
        self.MinkowskiInteraction = MinkowskiInteraction(self.filters, self.reduction)

        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

    def forward(self, query_features, corpus_features, batch_size):
        score_1 = self.TensorNetworkInteraction(query_features[-1], corpus_features[-1], batch_size)
        score_2 = self.MinkowskiInteraction(
                    torch.cat(query_features, dim=1),
                    torch.cat(corpus_features, dim=1)
                )
        score = self.alpha * score_1 + self.beta * score_2
        return score


class ERIC(torch.nn.Module):
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

        self.encoding_layer = EncodingLayer(input_dim, gnn_filters)
        self.interaction_layer = InteractionLayer(gnn_filters, tensor_neurons)

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        batch_size = len(graph_sizes)

        query_graphs, corpus_graphs = zip(*graphs)
        query_batch = Batch.from_data_list(query_graphs)
        corpus_batch = Batch.from_data_list(corpus_graphs)

        query_edge_index, corpus_edge_index = query_batch.edge_index, corpus_batch.edge_index
        query_node_features, corpus_node_features = query_batch.x, corpus_batch.x
        query_graph_idx, corpus_graph_idx = query_batch.batch, corpus_batch.batch

        # Encoding graph level features
        query_graph_features, query_node_features = self.encoding_layer(query_node_features, query_edge_index, query_graph_idx)
        corpus_graph_features, corpus_node_features = self.encoding_layer(corpus_node_features, corpus_edge_index, corpus_graph_idx)

        # Interaction
        score = self.interaction_layer(query_graph_features, corpus_graph_features)

        # Regularizer Term
        reg = self.encoding_layer.regularizer(query_graph_features, query_node_features, corpus_graph_features, corpus_node_features, query_graph_idx, corpus_graph_idx, batch_size)
        return score, reg
