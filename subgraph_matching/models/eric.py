import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
import torch.nn.functional as F   
from torch_geometric.nn.glob import global_add_pool
from torch_geometric.data import Batch

class TensorNetworkModule(torch.nn.Module):

    def __init__(self, tensor_neurons, filters):

        super(TensorNetworkModule, self).__init__()
        self.filters = filters
        self.tensor_neurons = tensor_neurons
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):

        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(
                self.filters, self.filters, self.tensor_neurons
            )
        )
        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(self.tensor_neurons, 2 * self.filters)
        )
        self.bias = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 1))

    def init_parameters(self):

        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):

        batch_size = len(embedding_1)
        scoring = torch.matmul(
            embedding_1, self.weight_matrix.view(self.filters, -1)
        )
        scoring = scoring.view(batch_size, self.filters, -1).permute([0, 2, 1])
        scoring = torch.matmul(
            scoring, embedding_2.view(batch_size, self.filters, 1)
        ).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(
            torch.mm(self.weight_matrix_block, torch.t(combined_representation))
        )
        scores = F.relu(scoring + block_scoring + self.bias.view(-1))
        return scores

class EncodingLayer:
    def __init__(self, input_dim, filters, tensor_neurons):
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
            self.mlp_list_inner.append(nn.Linear(self.filters[i], self.filters[i]))
            self.mlp_list_outer.append(nn.Linear(self.filters[i], self.filters[i]))


    def forward(self, x, edge_index, graph_idx, batch_size):
        size = batch[-1].item() + 1
        features_layer = []
        for i in range(self.num_filter):
            def conv(enc, x, edge_index):
                x = self.gnn_list[i](x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p = self.dropout, training=self.training)
                return x 
            x = conv(self.gnn_list[i], x, edge_index)
            x = F.relu(self.mlp_list_inner[i](x))
            x = model_utils.unsorted_segment_sum(x, graph_idx, batch_size)
            x = unsorted_segment_sum(feat, batch, size=size)
            x = F.relu(self.mlp_list_outer[i](x))
            features_layer.append(x)
        return torch.cat(features_layer, dim=1)


class InteractionLayer:
    def __init__(self, filters, tensor_neurons):
        super(InteractionLayer, self).__init__()
        self.NTN = TensorNetworkModule(tensor_neurons, filters)

    def forward(self, query_features, corpus_features):
        sim_rep = self.NTN(query_features, corpus_features)
        sim_score = torch.sigmoid(self.score_sim_layer(sim_rep).squeeze())
        self.score_sim_layer = nn.Sequential(
                                    nn.Linear(tensor_neurons, tensor_neurons),
                                    nn.ReLU(),
                                    nn.Linear(tensor_neurons, 1)
                                )

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

        self.filters = gnn_filters
        self.dropout = dropout
        self.device = device
        self.tensor_neurons = tensor_neurons
        self.num_filter = len(self.filters)

        self.encoding_layer = EncodingLayer(self.filters, self.tensor_neurons)
        self.interaction_layer = InteractionLayer(self.filters[-1], self.tensor_neurons)

        self.channel_dim = sum(self.filters)
        self.reduction = reduction
        self.conv_stack = nn.Sequential(
                            nn.Linear(self.channel_dim, self.channel_dim // self.reduction),
                            nn.ReLU(),
                            nn.Dropout(p = dropout),
                            nn.Linear(self.channel_dim // self.reduction, (self.channel_dim // self.reduction) ),
                            nn.Dropout(p = dropout),
                            nn.Tanh(),
                        )             

        self.score_layer = nn.Sequential(
                            nn.Linear((self.channel_dim // self.reduction) , 16),
                            nn.ReLU(),
                            nn.Linear(16 , 1)
                        )
        
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        batch_size = len(graph_sizes)

        query_graphs, corpus_graphs = zip(*graphs)
        query_batch = Batch.from_data_list(query_graphs)
        corpus_batch = Batch.from_data_list(corpus_graphs)

        query_edge_index, corpus_edge_index = query_batch.edge_index, corpus_batch.edge_index
        query_features, corpus_features = query_batch.x, corpus_batch.x
        query_graph_idx, corpus_graph_idx = query_batch.batch, corpus_batch.batch

        query_features = self.encoding_layer(query_features, query_edge_index, query_graph_idx)
        corpus_features = self.encoding_layer(corpus_features, corpus_edge_index, corpus_graph_idx)
        
        diff_rep = torch.exp(-torch.pow(query_features - corpus_features, 2))
        score_rep = self.conv_stack(diff_rep).squeeze()
        score = torch.sigmoid(self.score_layer(score_rep)).view(-1)

        sim_score = self.interaction_layer(query_features, corpus_features)
        comb_score = self.alpha * score + self.beta * sim_score

        return comb_score