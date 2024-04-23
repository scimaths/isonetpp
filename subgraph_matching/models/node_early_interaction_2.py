import torch
import functools
from utils import model_utils
from utils.tooling import ReadOnlyConfig
from subgraph_matching.models.gmn_iterative_refinement import GMNIterativeRefinement

class NodeEarlyInteraction2(GMNIterativeRefinement):
    def __init__(
        self,
        consistency_config: ReadOnlyConfig = None,
        **kwargs):
        super(NodeEarlyInteraction2, self).__init__(**kwargs)
        self.max_edge_set_size = kwargs['max_edge_set_size']
        self.consistency_config = consistency_config

    def get_interaction_features(self, node_features, transport_plan, graph_sizes, padded_node_indices):
        if transport_plan is None:
            return torch.zeros_like(node_features)

        stacked_feature_store_query, stacked_feature_store_corpus = model_utils.split_and_stack(
            node_features, graph_sizes, self.max_node_set_size
        )
        interleaved_node_features = model_utils.get_interaction_feature_store(
            transport_plan[0], stacked_feature_store_query, stacked_feature_store_corpus
        )
        interaction_features = interleaved_node_features[padded_node_indices]
        return interaction_features

    def forward_with_alignment(self, graphs, graph_sizes, graph_adj_matrices):
        query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = torch.tensor(query_sizes, device=self.device)
        corpus_sizes = torch.tensor(corpus_sizes, device=self.device)
        padded_node_indices = model_utils.get_padded_indices(graph_sizes, self.max_node_set_size, self.device)

        features_to_transport_plan = functools.partial(
            model_utils.features_to_transport_plan,
            query_sizes=query_sizes, corpus_sizes=corpus_sizes,
            graph_size_to_mask_map=self.graph_size_to_mask_map
        )

        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        num_nodes, node_feature_dim = encoded_node_features.shape

        transport_plan = None
        interaction_features = None

        for refine_idx in range(self.refinement_steps):

            node_features_enc, edge_features_enc = torch.clone(encoded_node_features), torch.clone(encoded_edge_features)
            interaction_features = torch.zeros(num_nodes, node_feature_dim, device=self.device)

            for prop_idx in range(1, self.propagation_steps + 1):

                node_features_enc = self.propagation_function(
                    from_idx, to_idx, node_features_enc, edge_features_enc, interaction_features
                )

                interaction_features = self.get_interaction_features(
                    node_features_enc, transport_plan, graph_sizes, padded_node_indices
                )

            stacked_feature_store_query, stacked_feature_store_corpus = model_utils.split_and_stack(
                node_features_enc, graph_sizes, self.max_node_set_size
            )

            transport_plan = features_to_transport_plan(
                stacked_feature_store_query, stacked_feature_store_corpus,
                preprocessor = self.interaction_alignment_preprocessor,
                alignment_function = self.interaction_alignment_function,
                what_for = 'interaction'
            )

        score, final_node_transport_plan = self.set_aligned_scoring(node_features_enc, graph_sizes, features_to_transport_plan)

        if self.consistency_config:
            combined_features = self.interaction_layer(
                torch.cat([node_features_enc, interaction_features], dim=-1)
            )

            messages = model_utils.propagation_messages(
                propagation_layer=self.prop_layer,
                node_features=combined_features,
                edge_features=edge_features_enc,
                from_idx=from_idx,
                to_idx=to_idx
            )

            consistency_score = model_utils.consistency_scoring(
                graphs=graphs,
                graph_sizes=graph_sizes,
                max_edge_set_size=self.max_edge_set_size,
                node_transport_plan=transport_plan[0],
                edge_features=messages,
                device=self.device,
                sinkhorn_config=self.sinkhorn_config
            ) * self.consistency_config.weight

            score += consistency_score

        return score, final_node_transport_plan
