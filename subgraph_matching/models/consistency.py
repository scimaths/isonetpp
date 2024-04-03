import torch
from utils import model_utils
from utils.tooling import ReadOnlyConfig


class Consistency(torch.nn.Module):
    def __init__(
        self,
        max_edge_set_size,
        edge_feature_dim,
        sinkhorn_config: ReadOnlyConfig,
        sinkhorn_feature_dim,
        consistency_config: ReadOnlyConfig,
        device,
    ):
        super(Consistency, self).__init__()

        self.max_edge_set_size = max_edge_set_size
        self.apply_transformation = consistency_config.apply_transformation
        self.consistency_weight = consistency_config.consistency_weight
        self.device = device

        self.graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
            max_set_size=max_edge_set_size, lateral_dim=sinkhorn_feature_dim, device=self.device
        )

        self.sinkhorn_config = sinkhorn_config
        if self.apply_transformation:
            self.sinkhorn_feature_layers = torch.nn.Sequential(
                torch.nn.Linear(edge_feature_dim, sinkhorn_feature_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(sinkhorn_feature_dim, sinkhorn_feature_dim),
            )

    def forward(self, graphs, graph_sizes, edge_features_enc, node_transport_plan):
        query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = torch.tensor(query_sizes, device=self.device)
        corpus_sizes = torch.tensor(corpus_sizes, device=self.device)

        _, _, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        paired_edge_counts = model_utils.get_paired_edge_counts(
            from_idx, to_idx, graph_idx, 2*len(graph_sizes)
        )

        stacked_features_query, stacked_features_corpus = model_utils.split_and_stack(
            edge_features_enc, paired_edge_counts, self.max_edge_set_size
        )

        if self.apply_transformation:
            transformed_features_query = self.sinkhorn_feature_layers(stacked_features_query)
            transformed_features_corpus = self.sinkhorn_feature_layers(stacked_features_corpus)

            def mask_graphs(features, graph_sizes):
                mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
                return mask * features
            stacked_features_query = mask_graphs(transformed_features_query, query_sizes)
            stacked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

        # Computation of edge transport plan
        straight_mapped_scores, cross_mapped_scores = model_utils.kronecker_product_on_nodes(
            node_transport_plan=node_transport_plan, from_idx=from_idx,
            to_idx=to_idx, paired_edge_counts=paired_edge_counts,
            graph_sizes=graph_sizes, max_edge_set_size=self.max_edge_set_size
        )

        edge_transport_plan = model_utils.sinkhorn_iters(
            log_alpha=straight_mapped_scores+cross_mapped_scores,
            device=self.device, **self.sinkhorn_config
        )

        return self.consistency_weight * model_utils.feature_alignment_score(
                query_features=stacked_features_query,
                corpus_features=stacked_features_corpus,
                transport_plan=edge_transport_plan
            )
