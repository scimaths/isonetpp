import torch
import torch.nn.functional as F
from utils import model_utils
from functools import partial
from subgraph_matching.models._template import AlignmentModel
from utils.tooling import ReadOnlyConfig
import GMN.graphembeddingnetwork as gmngen
from subgraph_matching.models.consistency import Consistency

# Interaction constants (wrt message-passing)
INTERACTION_PRE = 'pre'
INTERACTION_POST = 'post'

class EdgeEarlyInteraction1Baseline(torch.nn.Module):
    def __init__(
        self,
        max_node_set_size,
        max_edge_set_size,
        encoder_config: ReadOnlyConfig,
        propagation_layer_config: ReadOnlyConfig,
        propagation_steps,
        sinkhorn_config: ReadOnlyConfig,
        sinkhorn_feature_dim,
        device,
        interaction_when = INTERACTION_POST
    ):
        super(EdgeEarlyInteraction1Baseline, self).__init__()
        self.max_node_set_size = max_node_set_size
        self.max_edge_set_size = max_edge_set_size
        self.device = device

        self.graph_size_to_mask_map = {
            'interaction': model_utils.graph_size_to_mask_map(
                max_set_size=max_edge_set_size, lateral_dim=sinkhorn_feature_dim, device=self.device
            )
        }

        self.encoder = gmngen.GraphEncoder(**encoder_config)
        self.prop_layer = gmngen.GraphPropLayer(**propagation_layer_config)
        self.propagation_steps = propagation_steps

        self.message_dim = propagation_layer_config.edge_hidden_sizes[-1]
        interaction_input_dim = self.message_dim + encoder_config.edge_hidden_sizes[-1]
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

        self.interaction_alignment_function = lambda log_alpha, query_sizes, corpus_sizes: model_utils.sinkhorn_iters(
            log_alpha=log_alpha,  device=self.device, **self.sinkhorn_config
        )

        if interaction_when == INTERACTION_POST:
            self.propagation_function = self.propagation_step_with_post_interaction
        elif interaction_when == INTERACTION_PRE:
            self.propagation_function = self.propagation_step_with_pre_interaction

    def end_to_end_interaction_alignment(
        self, edge_features_enc, paired_edge_counts,
        features_to_transport_plan, padded_edge_indices
    ):
        stacked_features_query, stacked_features_corpus = model_utils.split_and_stack(
            edge_features_enc, paired_edge_counts, self.max_edge_set_size
        )

        transport_plan = features_to_transport_plan(
            stacked_features_query, stacked_features_corpus,
            preprocessor = self.sinkhorn_feature_layers,
            alignment_function = self.interaction_alignment_function,
            what_for = 'interaction'
        )

        interaction_features = model_utils.get_interaction_feature_store(
            transport_plan, stacked_features_query, stacked_features_corpus
        )[padded_edge_indices, :]

        return interaction_features

    def propagation_step_with_pre_interaction(
        self, prop_idx, from_idx, to_idx, paired_edge_counts,
        node_features_enc, edge_features_enc,
        features_to_transport_plan, padded_edge_indices
    ):
        if prop_idx == 0:
            interaction_features = torch.zeros_like(edge_features_enc)
        else:
            interaction_features = self.end_to_end_interaction_alignment(
                edge_features_enc, paired_edge_counts, features_to_transport_plan, padded_edge_indices
            )

        combined_features = self.interaction_encoder(
            torch.cat([edge_features_enc, interaction_features], dim=-1)
        )

        aggregated_messages = self.prop_layer._compute_aggregated_messages(
            node_features_enc, from_idx, to_idx, combined_features
        )
        node_features_enc = self.prop_layer._compute_node_update(node_features_enc, [aggregated_messages])

        edge_features_enc = model_utils.propagation_messages(
            propagation_layer=self.prop_layer,
            node_features=node_features_enc,
            edge_features=combined_features,
            from_idx=from_idx,
            to_idx=to_idx
        )

        return node_features_enc, edge_features_enc

    def propagation_step_with_post_interaction(
        self, prop_idx, from_idx, to_idx, paired_edge_counts,
        node_features_enc, edge_features_enc,
        features_to_transport_plan, padded_edge_indices
    ):
        interaction_features = self.end_to_end_interaction_alignment(
            edge_features_enc, paired_edge_counts, features_to_transport_plan, padded_edge_indices
        )

        aggregated_messages = self.prop_layer._compute_aggregated_messages(
            node_features_enc, from_idx, to_idx, edge_features_enc
        )
        node_features_enc = self.prop_layer._compute_node_update(
            node_features_enc, [aggregated_messages, aggregated_messages - interaction_features]
        )

        edge_features_enc = model_utils.propagation_messages(
            propagation_layer=self.prop_layer,
            node_features=node_features_enc,
            edge_features=edge_features_enc,
            from_idx=from_idx,
            to_idx=to_idx
        )
        edge_features_enc = self.interaction_encoder(
            torch.cat([edge_features_enc, interaction_features], dim=-1)
        )

        return node_features_enc, edge_features_enc

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = torch.tensor(query_sizes, device=self.device)
        corpus_sizes = torch.tensor(corpus_sizes, device=self.device)

        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        paired_edge_counts = model_utils.get_paired_edge_counts(
            from_idx, to_idx, graph_idx, 2*len(graph_sizes)
        )

        num_edges_query = [pair[0] for pair in paired_edge_counts]
        num_edges_corpus = [pair[1] for pair in paired_edge_counts]

        features_to_transport_plan = partial(
            model_utils.features_to_transport_plan,
            query_sizes=num_edges_query, corpus_sizes=num_edges_corpus,
            graph_size_to_mask_map=self.graph_size_to_mask_map
        )

        # Encode node and edge features
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)

        padded_edge_indices = model_utils.get_padded_indices(paired_edge_counts, self.max_edge_set_size, self.device)

        for prop_idx in range(1, self.propagation_steps + 1):

            # Message propagation on combined features
            node_features_enc, edge_features_enc = self.propagation_function(
                prop_idx, from_idx, to_idx, paired_edge_counts,
                node_features_enc, edge_features_enc,
                features_to_transport_plan, padded_edge_indices
            )

        stacked_features_query, stacked_features_corpus = model_utils.split_and_stack(
            edge_features_enc, paired_edge_counts, self.max_edge_set_size
        )

        transport_plan = features_to_transport_plan(
            stacked_features_query, stacked_features_corpus,
            preprocessor = self.sinkhorn_feature_layers,
            alignment_function = self.interaction_alignment_function,
            what_for = 'interaction'
        )

        score = model_utils.feature_alignment_score(stacked_features_query, stacked_features_corpus, transport_plan)

        return score
