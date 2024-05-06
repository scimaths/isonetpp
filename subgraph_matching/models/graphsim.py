import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
from subgraph_matching.models._template import EncodingLayer, InteractionLayer, EncodeThenInteractModel


class EncodingLayerGraphSim(EncodingLayer):
    def __init__(self, input_dim, gcn_size, dropout):
        super(EncodingLayerGraphSim, self).__init__()
        self.input_dim = input_dim
        self.gcn_size = gcn_size
        self.num_gcn_layers = len(self.gcn_size)
        self.dropout = dropout

        self.gcn_layers = torch.nn.ModuleList([])
        new_gcn_size = [self.input_dim] + self.gcn_size
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                pyg_nn.GCNConv(new_gcn_size[i], new_gcn_size[i+1])
            )

    def GCN_pass(self, features, edge_index):
        abstract_feature_matrices = []
        for i in range(self.num_gcn_layers-1):
            features = self.gcn_layers[i](features, edge_index)
            abstract_feature_matrices.append(features)
            features = F.relu(features)
            features = F.dropout(features, p=self.dropout, training=self.training)

        features = self.gcn_layers[-1](features, edge_index)
        abstract_feature_matrices.append(features)
        return abstract_feature_matrices

    def forward(self, graphs, graph_sizes, batch_size):
        graph_batch = Batch.from_data_list(graphs)
        features, edge_index, graph_idx = graph_batch.x, graph_batch.edge_index, graph_batch.batch

        features = self.GCN_pass(features, edge_index)
        features = [
            pad_sequence(torch.split(features[i], list(graph_sizes), dim=0), batch_first=True)
            for i in range(self.num_gcn_layers)
        ]
        return features


class InteractionLayerGraphSim(InteractionLayer):
    class CNNLayerV1(torch.nn.Module):
        def __init__(self, kernel_size, stride, in_channels, out_channels, num_similarity_matrices):
            super().__init__()
            self.stride = stride
            self.kernel_size = kernel_size
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.num_similarity_matrices = num_similarity_matrices
            padding_temp = (self.kernel_size - 1)//2
            if self.kernel_size%2 == 0:
                self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp+1, padding_temp, padding_temp+1))
            else:
                self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp, padding_temp, padding_temp))
            self.layers = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                                kernel_size=self.kernel_size, stride=stride) for i in range(num_similarity_matrices)])

        def forward(self, similarity_matrices_list):
            result = []
            for i in range(self.num_similarity_matrices):
                result.append(self.layers[i](self.padding(similarity_matrices_list[i])));
            return result


    class MaxPoolLayerV1(torch.nn.Module):
        def __init__(self, stride, pool_size, num_similarity_matrices):
            super().__init__()
            self.stride = stride
            self.pool_size = pool_size
            self.num_similarity_matrices = num_similarity_matrices
            padding_temp = (self.pool_size - 1)//2
            if self.pool_size%2 == 0:
                self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp+1, padding_temp, padding_temp+1))
            else:
                self.padding = torch.nn.ZeroPad2d((padding_temp, padding_temp, padding_temp, padding_temp))
            self.layers = torch.nn.ModuleList([torch.nn.MaxPool2d(kernel_size=self.pool_size, stride=stride) for i in range(num_similarity_matrices)])

        def forward(self, similarity_matrices_list):
            result = []
            for i in range(self.num_similarity_matrices):
                result.append(self.layers[i](self.padding(similarity_matrices_list[i])));
            return result


    def __init__(self, num_gcn_layers, conv_kernel_size, conv_pool_size, conv_out_channels, linear_size, dropout):
        super(InteractionLayerGraphSim, self).__init__()

        self.num_gcn_layers = num_gcn_layers
        self.conv_kernel_size = conv_kernel_size
        self.conv_pool_size = conv_pool_size
        self.conv_out_channels = conv_out_channels
        self.linear_size = linear_size
        self.dropout = dropout
        self.num_conv_layers = len(self.conv_kernel_size)
        self.num_linear_layers = len(self.linear_size)
        self.in_channels = 1

        self.conv_layers = torch.nn.ModuleList([])
        self.pool_layers = torch.nn.ModuleList([])
        for i in range(self.num_conv_layers):
            self.conv_layers.append(
                InteractionLayerGraphSim.CNNLayerV1(
                    kernel_size=self.conv_kernel_size[i],
                    stride=1,
                    in_channels=self.in_channels,
                    out_channels=self.conv_out_channels[i],
                    num_similarity_matrices=self.num_gcn_layers
                )
            )
            self.pool_layers.append(
                InteractionLayerGraphSim.MaxPoolLayerV1(
                    pool_size=self.conv_pool_size[i],
                    stride=self.conv_pool_size[i],
                    num_similarity_matrices=self.num_gcn_layers
                )
            )
            self.in_channels = self.conv_out_channels[i]

        self.linear_layers = torch.nn.ModuleList([])
        for i in range(self.num_linear_layers-1):
            self.linear_layers.append(
                torch.nn.Linear(self.linear_size[i],self.linear_size[i+1])
            )

        self.scoring_layer = torch.nn.Linear(self.linear_size[-1], 1)

    def Conv_pass(self, similarity_matrices_list):
        features = [_.unsqueeze(1) for _ in similarity_matrices_list]
        for i in range(self.num_conv_layers):
            features = self.conv_layers[i](features)
            features = [torch.relu(_)  for _ in features]
            features = self.pool_layers[i](features)
            features = [F.dropout(_, p=self.dropout, training=self.training) for _ in features]
        return features

    def linear_pass(self, features):
        for i in range(self.num_linear_layers-1):
            features = self.linear_layers[i](features)
            features = F.relu(features)
            features = F.dropout(features, p=self.dropout, training=self.training)
        return features

    def forward(self, query_features, corpus_features, batch_size):
        similarity_matrices_list = [
            torch.matmul(query_features[i], corpus_features[i].permute(0,2,1))
            for i in range(self.num_gcn_layers)
        ]
        features = torch.cat(
            self.Conv_pass(similarity_matrices_list), dim=1
        )
        features = features.view(-1, self.linear_size[0])
        features = self.linear_pass(features)
        score = self.scoring_layer(features).view(-1)
        return score


class GraphSim(EncodeThenInteractModel):
    def __init__(
        self,
        input_dim,
        max_node_set_size,
        max_edge_set_size,
        gcn_size,
        conv_kernel_size,
        conv_pool_size,
        conv_out_channels,
        linear_size,
        dropout,
        device
    ):
        super(GraphSim, self).__init__()
        self.encoding_layer = EncodingLayerGraphSim(input_dim, gcn_size, dropout)
        self.interaction_layer = InteractionLayerGraphSim(len(gcn_size), conv_kernel_size, conv_pool_size, conv_out_channels, linear_size, dropout)

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        batch_size = len(graph_sizes)
        query_graphs, corpus_graphs = zip(*graphs)
        query_sizes, corpus_sizes = zip(*graph_sizes)

        # Encoding graph level features
        query_graph_features = self.encoding_layer(query_graphs, query_sizes, batch_size)
        corpus_graph_features = self.encoding_layer(corpus_graphs, corpus_sizes, batch_size)

        # Interaction
        score = self.interaction_layer(query_graph_features, corpus_graph_features, batch_size)
        return score
