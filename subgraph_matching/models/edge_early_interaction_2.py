import torch
import torch.nn.functional as F
from utils import model_utils
from subgraph_matching.models._template import AlignmentModel
from utils.tooling import ReadOnlyConfig
import GMN.graphembeddingnetwork as gmngen
from subgraph_matching.models.consistency import Consistency


class EdgeEarlyInteraction2(torch.nn.Module):
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
        consistency_config: ReadOnlyConfig = None
    ):
        super(EdgeEarlyInteraction2, self).__init__()
        self.max_node_set_size = max_node_set_size
        self.max_edge_set_size = max_edge_set_size
        self.device = device

        self.graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
            max_set_size=max_edge_set_size, lateral_dim=sinkhorn_feature_dim, device=self.device
        )

        self.encoder = gmngen.GraphEncoder(**encoder_config)
        self.prop_layer = gmngen.GraphPropLayer(**propagation_layer_config)
        self.propagation_steps = propagation_steps
        self.time_update_steps = time_update_steps

        self.message_dim = propagation_layer_config.edge_hidden_sizes[-1]
        interaction_input_dim = self.message_dim + encoder_config.edge_feature_dim
        interaction_output_dim = propagation_layer_config.edge_embedding_dim
        self.interaction_encoder = torch.nn.Sequential(
            torch.nn.Linear(interaction_input_dim, interaction_input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(interaction_input_dim, interaction_output_dim)
        )

        self.sinkhorn_config = sinkhorn_config
        self.sinkhorn_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(self.message_dim, sinkhorn_feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(sinkhorn_feature_dim, sinkhorn_feature_dim)
        )

        self.consistency_config = consistency_config
        if self.consistency_config:

            self.node_sinkhorn_feature_layers = torch.nn.Sequential(
                torch.nn.Linear(propagation_layer_config.node_state_dim, sinkhorn_feature_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(sinkhorn_feature_dim, sinkhorn_feature_dim)
            )

            self.node_graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
                max_set_size=max_node_set_size, lateral_dim=sinkhorn_feature_dim, device=self.device
            )

            self.consistency_score = Consistency(
                self.max_edge_set_size,
                self.sinkhorn_config,
                consistency_config,
                self.device,
            )

    def get_interaction_features(self, edge_features, transport_plan, paired_edge_counts, padded_node_indices):
        if transport_plan is None:
            return torch.zeros_like(edge_features)

        stacked_feature_store_query, stacked_feature_store_corpus = model_utils.split_and_stack(
            edge_features, paired_edge_counts, self.max_node_set_size
        )
        interleaved_edge_features = model_utils.get_interaction_feature_store(
            transport_plan[0], stacked_feature_store_query, stacked_feature_store_corpus
        )
        interaction_features = interleaved_edge_features[padded_node_indices]
        return interaction_features

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = torch.tensor(query_sizes, device=self.device)
        corpus_sizes = torch.tensor(corpus_sizes, device=self.device)

        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        paired_edge_counts = model_utils.get_paired_edge_counts(
            from_idx, to_idx, graph_idx, 2*len(graph_sizes)
        )
        padded_edge_indices = model_utils.get_padded_indices(paired_edge_counts, self.max_edge_set_size, self.device)

        # Encode node and edge features
        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        num_edges, edge_feature_dim = encoded_edge_features.shape

        # Create feature stores, to be updated at every time index
        transport_plan = None
        interaction_features = None

        for _ in range(self.time_update_steps):

            node_features_enc, edge_features_enc = torch.clone(encoded_node_features), torch.clone(encoded_edge_features)
            interaction_features = torch.zeros(num_edges, self.message_dim, device=self.device)

            for prop_idx in range(1, self.propagation_steps + 1):

                combined_features = self.interaction_layer(
                    torch.cat([edge_features_enc, interaction_features], dim=-1)
                )

                node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, combined_features)

                edge_features_enc = model_utils.propagation_messages(
                    propagation_layer=self.prop_layer,
                    node_features=node_features_enc,
                    edge_features=combined_features,
                    from_idx=from_idx,
                    to_idx=to_idx
                )

                interaction_features = self.get_interaction_features(
                    edge_features_enc, transport_plan, graph_sizes, padded_edge_indices
                )

            stacked_feature_store_query, stacked_feature_store_corpus = model_utils.split_and_stack(
                edge_features_enc, paired_edge_counts, self.max_edge_set_size
            )

            transformed_features_query = self.sinkhorn_feature_layers(stacked_feature_store_query)
            transformed_features_corpus = self.sinkhorn_feature_layers(stacked_feature_store_corpus)

            def mask_graphs(features, graph_sizes):
                mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
                return mask * features
            num_edges_query = map(lambda pair: pair[0], paired_edge_counts)
            masked_features_query = mask_graphs(transformed_features_query, num_edges_query)
            num_edges_corpus = map(lambda pair: pair[1], paired_edge_counts)
            masked_features_corpus = mask_graphs(transformed_features_corpus, num_edges_corpus)

            sinkhorn_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
            transport_plan = model_utils.sinkhorn_iters(log_alpha=sinkhorn_input, device=self.device, **self.sinkhorn_config)

        score = model_utils.feature_alignment_score(stacked_feature_store_query, stacked_feature_store_corpus, transport_plan)

        return score
