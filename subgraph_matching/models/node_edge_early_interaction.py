import torch
import functools
from utils import model_utils
from utils.tooling import ReadOnlyConfig
from subgraph_matching.models.gmn_iterative_refinement import GMNIterativeRefinement


class NodeEdgeEarlyInteraction(GMNIterativeRefinement):
    def __init__(
        self,
        consistency_config: ReadOnlyConfig = None,
        **kwargs):
        super(NodeEdgeEarlyInteraction, self).__init__(**kwargs)
        self.max_edge_set_size = kwargs['max_edge_set_size']
        self.consistency_config = consistency_config
        propagation_layer_config = kwargs['propagation_layer_config']
        encoder_config = kwargs['encoder_config']

        self.message_dim = propagation_layer_config.edge_hidden_sizes[-1]
        interaction_input_dim = self.message_dim + encoder_config.edge_feature_dim
        interaction_output_dim = propagation_layer_config.edge_embedding_dim
        self.edge_interaction_layer = torch.nn.Sequential(
            torch.nn.Linear(interaction_input_dim, interaction_input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(interaction_input_dim, interaction_output_dim)
        )


    def propagation_step_with_pre_interaction(
        self, from_idx, to_idx, node_features_enc, edge_features_enc, interaction_features
    ):
        combined_features = self.interaction_layer(
            torch.cat([node_features_enc, interaction_features], dim=-1)
        )

        aggregated_messages = self.prop_layer._compute_aggregated_messages(
            combined_features, from_idx, to_idx, edge_features_enc
        )
        node_features_enc = self.prop_layer._compute_node_update(combined_features, [aggregated_messages])

        return node_features_enc


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

        paired_edge_counts = model_utils.get_paired_edge_counts(
            from_idx, to_idx, graph_idx, 2*len(graph_sizes)
        )
        padded_edge_indices = model_utils.get_padded_indices(paired_edge_counts, self.max_edge_set_size, self.device)

        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        num_nodes, node_feature_dim = encoded_node_features.shape
        num_edges, _ = encoded_edge_features.shape

        node_feature_store = torch.zeros(num_nodes, node_feature_dim * (self.propagation_steps + 1), device=self.device)
        updated_node_feature_store = torch.zeros_like(node_feature_store)

        edge_feature_store = torch.zeros(num_edges, self.message_dim * (self.propagation_steps + 1), device=self.device)
        updated_edge_feature_store = torch.zeros_like(edge_feature_store)


        for refine_idx in range(self.refinement_steps):
            node_features_enc, edge_features_enc = torch.clone(encoded_node_features), torch.clone(encoded_edge_features)

            interaction_features = node_feature_store[:, :node_feature_dim]

            for prop_idx in range(1, self.propagation_steps + 1):
                node_interaction_idx = node_feature_dim * prop_idx
                edge_interaction_idx = self.message_dim * prop_idx

                edge_interaction_features = edge_feature_store[:, edge_interaction_idx - self.message_dim : edge_interaction_idx]
                edge_combined_features = self.edge_interaction_layer(torch.cat([edge_features_enc, edge_interaction_features], dim=-1))

                node_features_enc = self.propagation_function(
                    from_idx, to_idx, node_features_enc, edge_combined_features, interaction_features
                )

                interaction_features = node_feature_store[:, node_interaction_idx : node_interaction_idx + node_feature_dim]
                combined_features = self.interaction_layer(
                    torch.cat([node_features_enc, interaction_features], dim=-1)
                )

                messages = model_utils.propagation_messages(
                    propagation_layer=self.prop_layer,
                    node_features=combined_features,
                    edge_features=edge_combined_features,
                    from_idx=from_idx,
                    to_idx=to_idx
                )

                updated_node_feature_store[:, node_interaction_idx : node_interaction_idx + node_feature_dim] = torch.clone(node_features_enc)
                updated_edge_feature_store[:, edge_interaction_idx : edge_interaction_idx + self.message_dim] = torch.clone(messages)

            stacked_feature_store_query, stacked_feature_store_corpus = model_utils.split_and_stack(
                updated_node_feature_store, graph_sizes, self.max_node_set_size
            )
            final_features_query = stacked_feature_store_query[:, :, -node_feature_dim:]
            final_features_corpus = stacked_feature_store_corpus[:, :, -node_feature_dim:]

            transport_plan = features_to_transport_plan(
                final_features_query, final_features_corpus,
                preprocessor = self.interaction_alignment_preprocessor,
                alignment_function = self.interaction_alignment_function,
                what_for = 'interaction'
            )

            interleaved_node_features = model_utils.get_interaction_feature_store(
                transport_plan[0], stacked_feature_store_query, stacked_feature_store_corpus
            )
            node_feature_store[:, node_feature_dim:] = interleaved_node_features[padded_node_indices, node_feature_dim:]

            print(transport_plan)
            straight_mapped_scores, cross_mapped_scores = model_utils.kronecker_product_on_nodes(
                node_transport_plan=node_transport_plan, from_idx=from_idx,
                to_idx=to_idx, paired_edge_counts=paired_edge_counts,
                graph_sizes=graph_sizes, max_edge_set_size=self.max_edge_set_size
            )

            edge_transport_plan = model_utils.sinkhorn_iters(
                log_alpha=straight_mapped_scores+cross_mapped_scores,
                device=self.device, **self.sinkhorn_config
            )

            edge_feature_store = torch.clone(updated_edge_feature_store)
            stacked_feature_store_query, stacked_feature_store_corpus = model_utils.split_and_stack(
                edge_feature_store, paired_edge_counts, self.max_edge_set_size
            )

            # Compute node transport plan
            final_edge_features_query = stacked_feature_store_query[:, :, -self.message_dim:]
            final_edge_features_corpus = stacked_feature_store_corpus[:, :, -self.message_dim:]

            # Compute interaction-based features
            interleaved_edge_features = model_utils.get_interaction_feature_store(
                edge_transport_plan, stacked_feature_store_query, stacked_feature_store_corpus
            )
            edge_feature_store[:, self.message_dim:] = interleaved_edge_features[padded_edge_indices, self.message_dim:]

        score = model_utils.feature_alignment_score(final_features_query, final_features_corpus, node_transport_plan)
        consistency = self.consistency_config.consistency_weight * model_utils.feature_alignment_score(final_edge_features_query, final_edge_features_corpus, edge_transport_plan)

        return score