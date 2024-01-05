import torch
import torch.nn.functional as F
from utils import model_utils
from utils.tooling import ReadOnlyConfig
import GMN.graphembeddingnetwork as gmngen


class NodeAlignNodeLossConsistency(torch.nn.Module):
    def __init__(
        self,
        max_node_set_size,
        max_edge_set_size,
        encoder_config: ReadOnlyConfig,
        propagation_layer_config: ReadOnlyConfig,
        propagation_steps,
        sinkhorn_config: ReadOnlyConfig,
        sinkhorn_feature_dim,
        consistency_weight,
        device,
    ):
        super(NodeAlignNodeLossConsistency, self).__init__()
        self.max_node_set_size = max_node_set_size
        self.max_edge_set_size = max_edge_set_size
        self.consistency_weight = consistency_weight
        self.device = device

        self.graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
            max_set_size=max_node_set_size, lateral_dim=sinkhorn_feature_dim, device=self.device,
        )

        self.encoder = gmngen.GraphEncoder(**encoder_config)
        self.prop_layer = gmngen.GraphPropLayer(**propagation_layer_config)
        self.propagation_steps = propagation_steps

        self.sinkhorn_config = sinkhorn_config
        self.node_sinkhorn_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(propagation_layer_config.node_state_dim, sinkhorn_feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(sinkhorn_feature_dim, sinkhorn_feature_dim),
        )

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = torch.tensor(query_sizes, device=self.device)
        corpus_sizes = torch.tensor(corpus_sizes, device=self.device)

        node_features, edge_features, from_idx, to_idx, graph_idx= model_utils.get_graph_features(graphs)

        # Propagation to compute node embeddings
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for _ in range(self.propagation_steps):
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, edge_features_enc)

        stacked_node_features_query, stacked_node_features_corpus = model_utils.split_and_stack(
            node_features_enc, graph_sizes, self.max_node_set_size
        )

        # Computation of node transport plan
        transformed_features_query = self.node_sinkhorn_feature_layers(stacked_node_features_query)
        transformed_features_corpus = self.node_sinkhorn_feature_layers(stacked_node_features_corpus)

        def mask_graphs(features, graph_sizes):
            mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
            return mask * features
        masked_features_query = mask_graphs(transformed_features_query, query_sizes)
        masked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

        node_sinkhorn_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
        node_transport_plan = model_utils.pytorch_sinkhorn_iters(
            log_alpha=node_sinkhorn_input, device=self.device, **self.sinkhorn_config
        )

        # Computation of edge embeddings
        edge_features_enc = model_utils.propagation_messages(
            propagation_layer=self.prop_layer,
            node_features=node_features_enc,
            edge_features=edge_features_enc,
            from_idx=from_idx,
            to_idx=to_idx
        )

        paired_edge_counts = model_utils.get_paired_edge_counts(
            from_idx, to_idx, graph_idx, 2*len(graph_sizes)
        )
        stacked_edge_features_query, stacked_edge_features_corpus = model_utils.split_and_stack(
            edge_features_enc, paired_edge_counts, self.max_edge_set_size
        )

        # Computation of edge transport plan
        straight_mapped_scores, cross_mapped_scores = model_utils.kronecker_product_on_nodes(
            node_transport_plan=node_transport_plan, from_idx=from_idx,
            to_idx=to_idx, paired_edge_counts=paired_edge_counts,
            graph_sizes=graph_sizes, max_edge_set_size=self.max_edge_set_size
        )
        edge_transport_plan = model_utils.pytorch_sinkhorn_iters(
            log_alpha=straight_mapped_scores+cross_mapped_scores,
            device=self.device, **self.sinkhorn_config
        )

        return torch.add(
            model_utils.feature_alignment_score(
                query_features=stacked_node_features_query,
                corpus_features=stacked_node_features_corpus,
                transport_plan=node_transport_plan
            ),
            self.consistency_weight * model_utils.feature_alignment_score(
                query_features=stacked_edge_features_query,
                corpus_features=stacked_edge_features_corpus,
                transport_plan=edge_transport_plan
            )
        )
