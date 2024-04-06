import torch
from utils import model_utils
from utils.tooling import ReadOnlyConfig
import GMN.graphembeddingnetwork as gmngen
from subgraph_matching.models._template import AlignmentModel
from subgraph_matching.models.consistency import Consistency


class NodeEdgeEarlyInteraction(torch.nn.Module):
    def __init__(
        self,
        max_node_set_size,
        max_edge_set_size,
        encoder_config: ReadOnlyConfig,
        propagation_layer_config: ReadOnlyConfig,
        propagation_steps,
        time_update_steps,
        sinkhorn_config: ReadOnlyConfig,
        sinkhorn_feature_dim,
        device,
        consistency_config: ReadOnlyConfig = None,
    ):
        super(NodeEdgeEarlyInteraction, self).__init__()
        self.max_node_set_size = max_node_set_size
        self.max_edge_set_size = max_edge_set_size
        self.device = device

        self.graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
            max_set_size=max_node_set_size, lateral_dim=sinkhorn_feature_dim, device=self.device
        )

        self.encoder = gmngen.GraphEncoder(**encoder_config)
        self.prop_layer = gmngen.GraphPropLayer(**propagation_layer_config)
        self.propagation_steps = propagation_steps
        self.time_update_steps = time_update_steps

        interaction_output_dim = propagation_layer_config.node_state_dim
        interaction_input_dim = 2 * interaction_output_dim
        self.node_interaction_encoder = torch.nn.Sequential(
            torch.nn.Linear(interaction_input_dim, interaction_input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(interaction_input_dim, interaction_output_dim)
        )

        self.sinkhorn_config = sinkhorn_config
        self.sinkhorn_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(propagation_layer_config.node_state_dim, sinkhorn_feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(sinkhorn_feature_dim, sinkhorn_feature_dim)
        )

        self.message_dim = propagation_layer_config.edge_hidden_sizes[-1]
        interaction_input_dim = self.message_dim + encoder_config.edge_feature_dim
        interaction_output_dim = propagation_layer_config.edge_embedding_dim
        self.edge_interaction_encoder = torch.nn.Sequential(
            torch.nn.Linear(interaction_input_dim, interaction_input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(interaction_input_dim, interaction_output_dim)
        )

        self.consistency_config = consistency_config
        if self.consistency_config:
            self.consistency_score = Consistency(
                self.max_edge_set_size,
                self.sinkhorn_config,
                consistency_config,
                self.device,
            )

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = torch.tensor(query_sizes, device=self.device)
        corpus_sizes = torch.tensor(corpus_sizes, device=self.device)

        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        paired_edge_counts = model_utils.get_paired_edge_counts(
            from_idx, to_idx, graph_idx, 2*len(graph_sizes)
        )

        # Encode node and edge features
        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        num_nodes, node_feature_dim = encoded_node_features.shape
        num_edges, edge_feature_dim = encoded_edge_features.shape

        # Create feature stores, to be updated at every time index
        node_feature_store = torch.zeros(num_nodes, node_feature_dim * (self.propagation_steps + 1), device=self.device)
        updated_node_feature_store = torch.zeros_like(node_feature_store)

        edge_feature_store = torch.zeros(num_edges, self.message_dim * (self.propagation_steps + 1), device=self.device)
        updated_edge_feature_store = torch.zeros_like(edge_feature_store)

        padded_node_indices = model_utils.get_padded_indices(graph_sizes, self.max_node_set_size, self.device)
        padded_edge_indices = model_utils.get_padded_indices(paired_edge_counts, self.max_edge_set_size, self.device)

        for _ in range(self.time_update_steps):
            node_features_enc, edge_features_enc = torch.clone(encoded_node_features), torch.clone(encoded_edge_features)

            interaction_features = node_feature_store[:, :node_feature_dim]
            node_combined_features = self.node_interaction_encoder(torch.cat([node_features_enc, interaction_features], dim=-1))

            for prop_idx in range(1, self.propagation_steps + 1):
                edge_interaction_idx = self.message_dim * prop_idx
                node_interaction_idx = node_feature_dim * prop_idx

                # Combine interaction features with edge features from previous propagation step
                interaction_features = edge_feature_store[:, edge_interaction_idx - self.message_dim : edge_interaction_idx]
                edge_combined_features = self.edge_interaction_encoder(torch.cat([edge_features_enc, interaction_features], dim=-1))

                # Message propagation on combined features
                node_features_enc = self.prop_layer(node_combined_features, from_idx, to_idx, edge_combined_features)

                # Combine interaction features with node features from previous propagation step
                interaction_features = node_feature_store[:, node_interaction_idx : node_interaction_idx + node_feature_dim]
                node_combined_features = self.node_interaction_encoder(torch.cat([node_features_enc, interaction_features], dim=-1))
                updated_node_feature_store[:, node_interaction_idx : node_interaction_idx + node_feature_dim] = torch.clone(node_features_enc)

                messages = model_utils.propagation_messages(
                    propagation_layer=self.prop_layer,
                    node_features=node_combined_features,
                    edge_features=edge_combined_features,
                    from_idx=from_idx,
                    to_idx=to_idx
                )
                updated_edge_feature_store[:, edge_interaction_idx : edge_interaction_idx + self.message_dim] = torch.clone(messages)

            node_feature_store = torch.clone(updated_node_feature_store)
            stacked_feature_store_query, stacked_feature_store_corpus = model_utils.split_and_stack(
                node_feature_store, graph_sizes, self.max_node_set_size
            )

            # Compute node transport plan
            final_features_query = stacked_feature_store_query[:, :, -node_feature_dim:]
            final_features_corpus = stacked_feature_store_corpus[:, :, -node_feature_dim:]

            transformed_features_query = self.sinkhorn_feature_layers(final_features_query)
            transformed_features_corpus = self.sinkhorn_feature_layers(final_features_corpus)

            def mask_graphs(features, graph_sizes):
                mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
                return mask * features
            masked_features_query = mask_graphs(transformed_features_query, query_sizes)
            masked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

            sinkhorn_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
            node_transport_plan = model_utils.sinkhorn_iters(log_alpha=sinkhorn_input, device=self.device, **self.sinkhorn_config)

            # Compute interaction-based features
            interleaved_node_features = model_utils.get_interaction_feature_store(
                node_transport_plan, stacked_feature_store_query, stacked_feature_store_corpus
            )
            node_feature_store[:, node_feature_dim:] = interleaved_node_features[padded_node_indices, node_feature_dim:]


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