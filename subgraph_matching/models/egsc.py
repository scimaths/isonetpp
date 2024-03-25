import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU
from torch_geometric.nn import GINConv
from torch_geometric.data import Batch
from utils import model_utils

class PoolingModule(nn.Module):
    def __init__(self, dim_size, reduction_ratio=4):
        super(PoolingModule, self).__init__()
        self.dim_size = dim_size
        self.mean_transform_layer = nn.Linear(dim_size, dim_size, bias=False)
        self.gating_layer = nn.Sequential(
            nn.Linear(dim_size,  dim_size // reduction_ratio),
            nn.ReLU(),
            nn.Linear(dim_size // reduction_ratio, dim_size),
            nn.Tanh()
        )
        self.final_layer = nn.Linear(dim_size,  dim_size)
        
    def forward(self, node_features, graph_idx, num_graphs):
        feature_coefs = self.gating_layer(node_features)
        scaled_node_features = (feature_coefs + 1) * node_features

        node_feature_mean = torch.div(
            model_utils.unsorted_segment_sum(scaled_node_features, graph_idx, num_graphs),
            model_utils.unsorted_segment_sum(torch.ones_like(node_features), graph_idx, num_graphs),
        )
        global_context = torch.tanh(self.mean_transform_layer(node_feature_mean)) 
        node_coefs = (scaled_node_features * global_context[graph_idx]).sum(dim=1).sigmoid()
        weighted_node_features = node_coefs.unsqueeze(-1) * scaled_node_features 

        return model_utils.unsorted_segment_sum(weighted_node_features, graph_idx, num_graphs)

class InteractionModule(nn.Module):
    def __init__(self, dim_size, reduction_ratio=4, output_dim=None):
        super(InteractionModule, self).__init__()
        self.channel_size = dim_size * 2
        self.gating_layer = nn.Sequential(
            nn.Linear(self.channel_size,  self.channel_size // reduction_ratio),
            nn.ReLU(),
            nn.Linear(self.channel_size // reduction_ratio, self.channel_size),
            nn.Sigmoid()
        )
        output_dim = output_dim or dim_size // 2
        self.final_layer = nn.Sequential(
            nn.Linear(self.channel_size,  self.channel_size),
            nn.ReLU(),
            nn.Linear(self.channel_size, output_dim),
            nn.ReLU()
        )

    def forward(self, query_embedding, corpus_embedding):
        combined_embedding = torch.cat((query_embedding, corpus_embedding), 1)
        feature_coefs = self.gating_layer(combined_embedding)
        scaled_embedding = (feature_coefs + 1) * combined_embedding
        return self.final_layer(scaled_embedding)

class ScoringModule(nn.Module):
    def __init__(self, dim_size, bottleneck_size, reduction_ratio=4):
        super(ScoringModule, self).__init__()
        self.dim_size = dim_size
        self.bottleneck_size = bottleneck_size
        self.gating_layer = nn.Sequential(
            nn.Linear(dim_size,  dim_size // reduction_ratio),
            nn.ReLU(),
            nn.Linear(dim_size // reduction_ratio, dim_size),
            nn.Sigmoid()
        )
        self.final_layer = nn.Sequential(
            nn.Linear(dim_size, bottleneck_size),
            nn.ReLU(),
            nn.Linear(bottleneck_size, 1),
        )

    def forward(self, combined_features):
        feature_coefs = self.gating_layer(combined_features)
        scaled_features = (feature_coefs + 1) * combined_features
        return self.final_layer(scaled_features).squeeze()

class EGSC_ScoringHead(nn.Module):
    def __init__(self, dim_size, bottleneck_size, reduction_ratio=4):
        super(EGSC_ScoringHead, self).__init__()
        self.pooling_layer = PoolingModule(dim_size, reduction_ratio)
        self.interaction_layer = InteractionModule(dim_size, reduction_ratio, output_dim=dim_size)
        self.scoring_layer = ScoringModule(dim_size, bottleneck_size, reduction_ratio)

    def forward(self, node_features, graph_idx, num_graphs):
        pooled_graph_features = self.pooling_layer(node_features, graph_idx, num_graphs)
        query_features, corpus_features = pooled_graph_features[0::2], pooled_graph_features[1::2]
        combined_features = self.interaction_layer(query_features, corpus_features)
        return self.scoring_layer(combined_features)

class EGSC(nn.Module):
    def __init__(
        self,
        max_node_set_size,
        max_edge_set_size,
        input_feature_dim,
        conv_filters,
        bottleneck_size,
        dropout,
        device,
        reduction_ratio=4,
    ):
        super(EGSC, self).__init__()
        self.device = device
        self.num_conv_layers = len(conv_filters)
        self.conv_filters = conv_filters
        self.dropout = dropout

        self.aggregated_feature_dim = sum(conv_filters) // 2

        self.gin_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()

        gin_feature_dim_sequence = [input_feature_dim] + self.conv_filters
        for idx in range(self.num_conv_layers):
            # GIN layers
            input_dim, output_dim = gin_feature_dim_sequence[idx : idx + 2]
            mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )
            self.gin_layers.append(GINConv(mlp, train_eps=True))

            # Pooling and Interaction layers
            self.pooling_layers.append(PoolingModule(dim_size=output_dim, reduction_ratio=reduction_ratio))
            self.interaction_layers.append(InteractionModule(dim_size=output_dim, reduction_ratio=reduction_ratio))

        self.scoring_layer = ScoringModule(
            dim_size=self.aggregated_feature_dim,
            bottleneck_size=bottleneck_size,
            reduction_ratio=reduction_ratio
        )

    def gin_forward_pass(self, layer_idx, features, edge_index):
        features = self.gin_layers[layer_idx](features, edge_index)
        features = F.relu(features)
        return F.dropout(features, p=self.dropout)

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        batch_size = len(graph_sizes)

        query_graphs, corpus_graphs = zip(*graphs)
        query_batch = Batch.from_data_list(query_graphs)
        corpus_batch = Batch.from_data_list(corpus_graphs)

        query_edge_index, corpus_edge_index = query_batch.edge_index, corpus_batch.edge_index
        query_features, corpus_features = query_batch.x, corpus_batch.x
        query_graph_idx, corpus_graph_idx = query_batch.batch, corpus_batch.batch

        combined_features = []
        for layer_idx in range(self.num_conv_layers):
            query_features = self.gin_forward_pass(
                layer_idx=layer_idx, features=query_features, edge_index=query_edge_index
            )
            pooled_query_features = self.pooling_layers[layer_idx](
                query_features, query_graph_idx, batch_size
            )

            corpus_features = self.gin_forward_pass(
                layer_idx=layer_idx, features=corpus_features, edge_index=corpus_edge_index
            )
            pooled_corpus_features = self.pooling_layers[layer_idx](
                corpus_features, corpus_graph_idx, batch_size
            )

            combined_features.append(
                self.interaction_layers[layer_idx](pooled_query_features, pooled_corpus_features)
            )
        combined_features = torch.cat(combined_features, dim=1)

        return self.scoring_layer(combined_features)