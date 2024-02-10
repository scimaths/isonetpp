import torch
from typing import Optional
from utils import model_utils
from functools import partial
from utils.tooling import ReadOnlyConfig
import GMN.graphembeddingnetwork as gmngen
from subgraph_matching.models._template import AlignmentModel

# Alignment preprocessing constants
LRL = 'lrl'
IDENTITY = 'identity'
POSSIBLE_ALIGNMENT_PREPROCESSOR_TYPES = [LRL, IDENTITY]

# Alignment constants
SINKHORN = 'sinkhorn'
ATTENTION = 'attention'
MASKED_ATTENTION = 'masked_attention'
POSSIBLE_ALIGNMENTS = [ATTENTION, MASKED_ATTENTION, SINKHORN, None]

# Scoring constants
AGGREGATED = 'aggregated'
SET_ALIGNED = 'set_aligned'
POSSIBLE_SCORINGS = [AGGREGATED, SET_ALIGNED]

class GMNBaseline(AlignmentModel):
    def __init__(
        self,
        max_node_set_size,
        max_edge_set_size,
        encoder_config: ReadOnlyConfig,
        propagation_layer_config: ReadOnlyConfig,
        propagation_steps,
        device,
        # Arguments to manage scoring-time alignment
        scoring: str = AGGREGATED, # one of 'aggregated', 'set_aligned'
        aggregator_config: Optional[ReadOnlyConfig] = None,
        scoring_alignment: Optional[str] = None, # one of 'attention', 'sinkhorn' or None
        scoring_alignment_preprocessor_type: str = IDENTITY, # one of 'lrl' or 'identity'
        scoring_alignment_feature_dim: Optional[int] = None,
        # Use scoring arguments for interaction
        unify_scoring_and_interaction: bool = False,
        # Arguments to manage interaction-time alignment
        interaction_alignment: Optional[str] = ATTENTION, # one of 'attention' or 'sinkhorn' if not unified, else None
        interaction_alignment_preprocessor_type: str = IDENTITY, # one of 'lrl' or 'identity'
        interaction_alignment_feature_dim: Optional[int] = None,
        # Arguments to manage alignment configs - shared if `scoring_alignment` and `interaction_alignment` are identical
        sinkhorn_config: Optional[ReadOnlyConfig] = None,
        attention_config: Optional[ReadOnlyConfig] = None,
    ):
        super(GMNBaseline, self).__init__()

        #########################################
        # CONSTRAINTS for scoring
        assert (scoring in POSSIBLE_SCORINGS), f"`scoring` must be one of {POSSIBLE_SCORINGS}, found {scoring}"
        assert (scoring_alignment in POSSIBLE_ALIGNMENTS), (
            f"`scoring_alignment` must be one of {POSSIBLE_ALIGNMENTS}, found {scoring_alignment}"
        )
        # ensure aggregator_config is present when needed and not when not
        assert (scoring != AGGREGATED) ^ (aggregator_config is not None), (
            "`aggregator_config` should not be None iff aggregated scoring is used"
        )
        # set_aligned scoring should use some non-None alignment
        assert (scoring == AGGREGATED) ^ (scoring_alignment is not None), (
            "`scoring_alignment` should be None iff aggregated scoring is used"
        )
        # require feature_dim for LRL preprocessing
        assert (scoring_alignment_preprocessor_type == LRL) ^ (scoring_alignment_feature_dim is None), (
            "`scoring_alignment_feature_dim` should be non-zero iff LRL preprocessing is used"
        )
        # ensure no extra params if aggregated
        assert (scoring != AGGREGATED) or (scoring_alignment_preprocessor_type == IDENTITY), (
            "aggregated scoring must have identity preprocessor to prevent extra parameters"
        )
        self.scoring = scoring
        self.aggregator_config = aggregator_config
        self.scoring_alignment_type = scoring_alignment
        self.scoring_alignment_feature_dim = scoring_alignment_feature_dim
        self.scoring_alignment_preprocessor_type = scoring_alignment_preprocessor_type

        #########################################
        # CONSTRAINTS for interaction
        assert not(unify_scoring_and_interaction) or (scoring != AGGREGATED), (
            "Can't unify with aggregated scoring"
        )
        assert unify_scoring_and_interaction ^ (interaction_alignment in POSSIBLE_ALIGNMENTS and interaction_alignment is not None), (
            f"`interaction_alignment` must be one of {POSSIBLE_ALIGNMENTS} and not None iff not-unified else None, found {interaction_alignment}"
        )
        # require feature_dim for LRL preprocessing
        assert (interaction_alignment_preprocessor_type == LRL) ^ (interaction_alignment_feature_dim is None), (
            "`interaction_alignment_feature_dim` should be non-zero iff LRL preprocessing is used"
        )
        self.unify_scoring_and_interaction = unify_scoring_and_interaction
        self.interaction_alignment_type = interaction_alignment
        self.interaction_alignment_feature_dim = interaction_alignment_feature_dim
        self.interaction_alignment_preprocessor_type = interaction_alignment_preprocessor_type

        #########################################
        # CONSTRAINTS for configs
        alignment_types_used = [scoring_alignment, interaction_alignment]
        assert (sinkhorn_config is None) ^ SINKHORN in alignment_types_used, (
            "`sinkhorn_config` was specified but it was not used in scoring or interaction or vice-versa"
        )
        assert (attention_config is None) ^ (
            ATTENTION in alignment_types_used or MASKED_ATTENTION in alignment_types_used
        ), (
            "`attention_config` was specified but it was not used in scoring or interaction or vice-versa"
        )
        self.sinkhorn_config = sinkhorn_config
        self.attention_config = attention_config

        #########################################
        # Actual implementation begins
        self.max_node_set_size = max_node_set_size
        self.device = device

        # Handle common layers
        self.encoder = gmngen.GraphEncoder(**encoder_config)
        self.prop_layer = gmngen.GraphPropLayer(**propagation_layer_config)
        self.propagation_steps = propagation_steps

        # Handle unification of graph_size_to_mask_map
        self.graph_size_to_mask_map = {
            'scoring': model_utils.graph_size_to_mask_map(
                max_set_size=max_node_set_size, lateral_dim=scoring_alignment_feature_dim, device=self.device
            )
        }
        self.graph_size_to_mask_map['interaction'] = self.graph_size_to_mask_map['scoring'] if unify_scoring_and_interaction else (
            model_utils.graph_size_to_mask_map(
                max_set_size=max_node_set_size, lateral_dim=interaction_alignment_feature_dim, device=self.device
            )
        )

        # Setup scoring and interaction layer
        self.setup_scoring(propagation_layer_config.node_state_dim)
        self.setup_interaction(propagation_layer_config.node_state_dim)

    def get_alignment_preprocessor(self, preprocessor_type, preprocessor_feature_dim, node_state_dim):
        if preprocessor_type == IDENTITY:
            return lambda x: x
        elif preprocessor_type == LRL:
            return torch.nn.Sequential(
                torch.nn.Linear(node_state_dim, preprocessor_feature_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(preprocessor_feature_dim, preprocessor_feature_dim)
            )
        else:
            raise NotImplementedError(f"preprocessor is implemented only for these modes - {POSSIBLE_ALIGNMENT_PREPROCESSOR_TYPES}")

    def get_alignment_function(self, alignment_type):
        if alignment_type == ATTENTION:
            return lambda log_alpha, query_sizes, corpus_sizes: model_utils.attention(log_alpha=log_alpha, **self.attention_config)
        elif alignment_type == MASKED_ATTENTION:
            return lambda log_alpha, query_sizes, corpus_sizes: model_utils.masked_attention(
                log_alpha=log_alpha, query_sizes=query_sizes, corpus_sizes=corpus_sizes, **self.attention_config
            )
        elif alignment_type == SINKHORN:
            return lambda log_alpha, query_sizes, corpus_sizes: model_utils.sinkhorn_iters(
                log_alpha=log_alpha,  device=self.device, **self.sinkhorn_config
            )

    def setup_interaction(self, node_state_dim):
        if self.unify_scoring_and_interaction:
            self.interaction_alignment_preprocessor = self.scoring_alignment_preprocessor
            self.interaction_alignment_function = self.scoring_alignment_function
        else:
            self.interaction_alignment_preprocessor = self.get_alignment_preprocessor(
                preprocessor_type = self.interaction_alignment_preprocessor_type,
                preprocessor_feature_dim = self.interaction_alignment_feature_dim,
                node_state_dim = node_state_dim
            )
            self.interaction_alignment_function = self.get_alignment_function(alignment_type=self.interaction_alignment_type)

    def setup_scoring(self, node_state_dim):
        if self.scoring == AGGREGATED:
            self.aggregator = gmngen.GraphAggregator(**self.aggregator_config)
        elif self.scoring == SET_ALIGNED:
            self.scoring_alignment_preprocessor = self.get_alignment_preprocessor(
                preprocessor_type = self.scoring_alignment_preprocessor_type,
                preprocessor_feature_dim = self.scoring_alignment_feature_dim,
                node_state_dim = node_state_dim
            )
            self.scoring_alignment_function = self.get_alignment_function(alignment_type=self.scoring_alignment_type)
        else:
            raise NotImplementedError(f"scoring is implemented only for these modes - {POSSIBLE_SCORINGS}")

    def forward_with_alignment(self, graphs, graph_sizes, graph_adj_matrices):
        query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = torch.tensor(query_sizes, device=self.device)
        corpus_sizes = torch.tensor(corpus_sizes, device=self.device)

        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for _ in range(self.propagation_steps) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, edge_features_enc)

        ############################## SCORING ##############################
        if self.scoring == AGGREGATED:
            graph_vectors = self.aggregator(node_features_enc, graph_idx, 2 * len(graph_sizes))
            graph_vector_dim = graph_vectors.shape[-1]
            reshaped_graph_vectors = graph_vectors.reshape(-1, graph_vector_dim * 2)
            query_graph_vectors = reshaped_graph_vectors[:, :graph_vector_dim]
            corpus_graph_vectors = reshaped_graph_vectors[:, graph_vector_dim:]

            return -torch.sum(
                torch.nn.functional.relu(query_graph_vectors - corpus_graph_vectors),
                dim=-1
            ), []

        elif self.scoring == SET_ALIGNED:
            stacked_features_query, stacked_features_corpus = model_utils.split_and_stack(
                node_features_enc, graph_sizes, self.max_node_set_size
            )
            transformed_features_query = self.interaction_alignment_preprocessor(stacked_features_query)
            transformed_features_corpus = self.interaction_alignment_preprocessor(stacked_features_corpus)

            def mask_graphs(features, graph_sizes):
                mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
                return mask * features
            masked_features_query = mask_graphs(transformed_features_query, query_sizes)
            masked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

            log_alpha = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
            transport_plan = self.interaction_alignment_function(log_alpha=log_alpha, query_sizes=query_sizes, corpus_sizes=corpus_sizes)
        
            return model_utils.feature_alignment_score(
                stacked_features_query, stacked_features_corpus, transport_plan
            ), [transport_plan, transport_plan.transpose(-1, -2)]
