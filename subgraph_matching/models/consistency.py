import torch
from utils import model_utils
from utils.tooling import ReadOnlyConfig


class Consistency(torch.nn.Module):
    def __init__(
        self,
        max_edge_set_size,
        sinkhorn_config: ReadOnlyConfig,
        consistency_config: ReadOnlyConfig,
        device
    ):
        super(Consistency, self).__init__()

        self.max_edge_set_size = max_edge_set_size
        self.consistency_weight = consistency_config.consistency_weight
        self.device = device
        self.sinkhorn_config = sinkhorn_config

    def forward(self, graphs, graph_sizes, messages, node_transport_plan):

        _, _, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        paired_edge_counts = model_utils.get_paired_edge_counts(
            from_idx, to_idx, graph_idx, 2*len(graph_sizes)
        )

        stacked_features_query, stacked_features_corpus = model_utils.split_and_stack(
            messages, paired_edge_counts, self.max_edge_set_size
        )

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