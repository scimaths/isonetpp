import torch
import functools
from typing import Optional
from utils import model_utils
from functools import partial
from utils.tooling import ReadOnlyConfig
import GMN.graphembeddingnetwork as gmngen
import GMN.graphmatchingnetwork as gmngmn
from subgraph_matching.models._template import AlignmentModel
from subgraph_matching.modules import neural_tensor_network as ntn

# Alignment preprocessing constants
LRL = 'lrl'
IDENTITY = 'identity'
HINGE = 'hinge'
POSSIBLE_ALIGNMENT_PREPROCESSOR_TYPES = [LRL, IDENTITY, HINGE]

# Alignment constants
SINKHORN = 'sinkhorn'
ATTENTION = 'attention'
MASKED_ATTENTION = 'masked_attention'
POSSIBLE_ALIGNMENTS = [ATTENTION, MASKED_ATTENTION, SINKHORN, None]

# Scoring constants
AGGREGATED = 'aggregated'
SET_ALIGNED = 'set_aligned'
NEURAL = 'neural'
NTN = 'ntn'
POSSIBLE_SCORINGS = [AGGREGATED, SET_ALIGNED, NEURAL, NTN]

# Interaction constants (wrt message-passing)
INTERACTION_NEVER = 'never'
INTERACTION_PRE = 'pre'
INTERACTION_POST = 'post'
INTERACTION_MSG_ONLY = 'msg_passing_only'
INTERACTION_UPD_ONLY = 'update_only'

class GMNEdgeBaseline(AlignmentModel):
    def __init__(
        self,
        max_node_set_size,
        max_edge_set_size,
        encoder_config: ReadOnlyConfig,
        propagation_layer_config: ReadOnlyConfig,
        propagation_steps,
        device,
        alignment_feature_dim: Optional[int] = None,
        # Arguments to manage scoring-time alignment
        scoring: str = AGGREGATED, # one of 'aggregated', 'set_aligned', 'neural', 'ntn'
        ntn_config: Optional[ReadOnlyConfig] = None,
        aggregator_config: Optional[ReadOnlyConfig] = None,
        scoring_alignment: Optional[str] = None, # one of 'attention', 'sinkhorn' or None
        scoring_alignment_preprocessor_type: str = IDENTITY, # one of 'lrl', 'hinge' or 'identity'
        # Use scoring arguments for interaction
        unify_scoring_and_interaction_preprocessor: bool = False,
        # Arguments to manage interaction-time alignment
        interaction_alignment: Optional[str] = None, # one of 'attention' or 'sinkhorn' if not unified, else None
        interaction_alignment_preprocessor_type: str = IDENTITY, # one of 'lrl', 'hinge' or 'identity'
        # Arguments to manage alignment configs - shared if `scoring_alignment` and `interaction_alignment` are identical
        sinkhorn_config: Optional[ReadOnlyConfig] = None,
        attention_config: Optional[ReadOnlyConfig] = None,
        # Arguments for when of interaction
        interaction_when: str = INTERACTION_POST,
    ):
        super(GMNEdgeBaseline, self).__init__()

        #########################################
        # CONSTRAINTS for scoring
        assert (scoring in POSSIBLE_SCORINGS), f"`scoring` must be one of {POSSIBLE_SCORINGS}, found {scoring}"
        assert (scoring_alignment in POSSIBLE_ALIGNMENTS), (
            f"`scoring_alignment` must be one of {POSSIBLE_ALIGNMENTS}, found {scoring_alignment}"
        )
        # ensure aggregator_config is present when needed and not when not
        assert (scoring not in [AGGREGATED, NEURAL, NTN]) ^ (aggregator_config is not None), (
            "`aggregator_config` should not be None iff aggregated/neural/ntn scoring is used"
        )
        # set_aligned scoring should use some non-None alignment
        assert (scoring != SET_ALIGNED) ^ (scoring_alignment is not None), (
            "`scoring_alignment` should be None iff set-aligned scoring is not used"
        )
        # require feature_dim for LRL preprocessing
        assert (scoring_alignment_preprocessor_type != LRL) or (alignment_feature_dim is not None), (
            "`alignment_feature_dim` should be non-zero if LRL preprocessing is used in scoring"
        )
        # ensure no extra params if aggregated
        assert (scoring != AGGREGATED) or (scoring_alignment_preprocessor_type == IDENTITY), (
            "aggregated scoring must have identity preprocessor to prevent extra parameters"
        )
        self.scoring = scoring
        self.aggregator_config = aggregator_config
        self.ntn_config = ntn_config
        self.scoring_alignment_type = scoring_alignment
        self.alignment_feature_dim = alignment_feature_dim
        self.scoring_alignment_preprocessor_type = scoring_alignment_preprocessor_type

        #########################################
        # CONSTRAINTS for interaction
        # unification of interaction and scoring
        assert not(unify_scoring_and_interaction_preprocessor) or (scoring not in [AGGREGATED, NEURAL, NTN]), (
            "Can't unify with aggregated/neural/ntn scoring"
        )
        assert not(unify_scoring_and_interaction_preprocessor) or (
            interaction_alignment_preprocessor_type == scoring_alignment_preprocessor_type
        ), "Unification requires both preprocessors to be identical"
        # require feature_dim for LRL preprocessing
        assert (interaction_alignment_preprocessor_type != LRL) or (alignment_feature_dim is not None), (
            "`alignment_feature_dim` should be non-zero if LRL preprocessing is used in interaction"
        )
        # require interaction is pre/post
        assert interaction_when in [INTERACTION_NEVER, INTERACTION_POST], (
            "`interaction_when` must be one of `none`/`post`"
        )
        assert (interaction_when, propagation_layer_config.prop_type) in [
            (INTERACTION_NEVER, 'embedding'),
            (INTERACTION_POST, 'embedding'),
        ]
        # require essential encoder output dimension for edge interaction
        assert propagation_layer_config.edge_hidden_sizes[-1] == encoder_config.edge_hidden_sizes[-1], (
            "if message_dim != dim of edge encoding, can't use initial edge encoding directly"
        )

        #########################################
        # CONSTRAINTS for no interaction
        assert (interaction_when != INTERACTION_NEVER or scoring == SET_ALIGNED) or (
            sinkhorn_config is None and attention_config is None
        ), "For no interaction and non-set-aligned scoring, no alignment procedure is required"

        assert (interaction_alignment is None) == (interaction_when == INTERACTION_NEVER), (
            "Alignment kind and stage of interaction should be consistent for the no interaction case"
        )

        assert (interaction_when != INTERACTION_NEVER) or (not unify_scoring_and_interaction_preprocessor), (
            "Unification of LRL preprocessors not possible in the case of no interaction"
        )

        self.unify_scoring_and_interaction_preprocessor = unify_scoring_and_interaction_preprocessor
        self.interaction_alignment_type = interaction_alignment
        self.interaction_alignment_preprocessor_type = interaction_alignment_preprocessor_type
        self.interaction_when = interaction_when

        #########################################
        # CONSTRAINTS for configs
        alignment_types_used = [scoring_alignment, interaction_alignment]
        assert (sinkhorn_config is None) ^ (SINKHORN in alignment_types_used), (
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
        self.device = device

        # Handle common layers
        self.encoder = gmngen.GraphEncoder(**encoder_config)
        self.prop_layer = gmngmn.GraphPropLayer(**propagation_layer_config)

        # Propagation params and function
        self.propagation_steps = propagation_steps
        self.message_dim = propagation_layer_config.edge_hidden_sizes[-1]
        self.edge_state_dim = propagation_layer_config.edge_embedding_dim
        self.set_max_edge_set_size(max_edge_set_size)

        if self.interaction_when == INTERACTION_POST:
            self.propagation_function = self.propagation_step_with_post_interaction
        elif self.interaction_when == INTERACTION_NEVER:
            self.propagation_function = self.propagation_step_without_interaction

        # Setup scoring and interaction layer
        # Order is reversed compared to gmn_baseline.py to match
        # the order of network initializations in edge_early_interaction_baseline_1.py
        self.setup_interaction(self.edge_state_dim)
        self.setup_scoring(self.edge_state_dim)

    def set_max_edge_set_size(self, max_edge_set_size):
        self.max_edge_set_size = max_edge_set_size

        # Handle unification of graph_size_to_mask_map
        self.graph_size_to_mask_map = {
            key: model_utils.graph_size_to_mask_map(
                max_set_size = max_edge_set_size, device=self.device,
                lateral_dim = self.alignment_feature_dim if preprocessor_type == LRL else self.edge_state_dim
            ) for (key, preprocessor_type) in [
                ('scoring', self.scoring_alignment_preprocessor_type),
                ('interaction', self.interaction_alignment_preprocessor_type),
            ]
        }

    def get_alignment_preprocessor(self, preprocessor_type, preprocessor_feature_dim, edge_state_dim):
        if preprocessor_type == IDENTITY:
            return lambda x: x
        elif preprocessor_type == LRL:
            return torch.nn.Sequential(
                torch.nn.Linear(edge_state_dim, preprocessor_feature_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(preprocessor_feature_dim, preprocessor_feature_dim)
            )
        elif preprocessor_type == HINGE:
            return HINGE
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
            return_self_and_transpose = lambda x: (x, x.transpose(-1, -2))
            return lambda log_alpha, query_sizes, corpus_sizes: return_self_and_transpose(model_utils.sinkhorn_iters(
                log_alpha=log_alpha,  device=self.device, **self.sinkhorn_config
            ))

    def setup_interaction(self, edge_state_dim):
        if self.interaction_when == INTERACTION_NEVER: return

        self.interaction_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.message_dim * 2, self.message_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.message_dim * 2, self.edge_state_dim)
        )
        self.interaction_alignment_preprocessor = self.get_alignment_preprocessor(
            preprocessor_type = self.interaction_alignment_preprocessor_type,
            preprocessor_feature_dim = self.alignment_feature_dim,
            edge_state_dim = edge_state_dim
        )
        self.interaction_alignment_function = self.get_alignment_function(alignment_type=self.interaction_alignment_type)

    def setup_scoring(self, edge_state_dim):
        if self.scoring in [AGGREGATED, NEURAL, NTN]:
            self.aggregator = gmngen.GraphAggregator(**self.aggregator_config)

            if self.scoring == NEURAL:
                graph_vector_dim = self.aggregator.get_output_dim()
                self.scoring_mlp = torch.nn.Sequential(
                    torch.nn.Linear(2 * graph_vector_dim, graph_vector_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(graph_vector_dim, 1)
                )
            elif self.scoring == NTN:
                self.ntn_layer = ntn.NeuralTensorNetwork(**self.ntn_config)

        elif self.scoring == SET_ALIGNED:
            if self.unify_scoring_and_interaction_preprocessor:
                self.scoring_alignment_preprocessor = self.interaction_alignment_preprocessor
            else:
                self.scoring_alignment_preprocessor = self.get_alignment_preprocessor(
                    preprocessor_type = self.scoring_alignment_preprocessor_type,
                    preprocessor_feature_dim = self.alignment_feature_dim,
                    edge_state_dim = edge_state_dim
                )

            self.scoring_alignment_function = self.get_alignment_function(alignment_type=self.scoring_alignment_type)

    def end_to_end_interaction_alignment(
        self, edge_features_enc, paired_edge_counts,
        features_to_transport_plan, padded_edge_indices
    ):
        stacked_features_query, stacked_features_corpus = model_utils.split_and_stack(
            edge_features_enc, paired_edge_counts, self.max_edge_set_size
        )

        transport_plan = features_to_transport_plan(
            stacked_features_query, stacked_features_corpus,
            preprocessor = self.interaction_alignment_preprocessor,
            alignment_function = self.interaction_alignment_function,
            what_for = 'interaction'
        )

        interaction_features = model_utils.get_interaction_feature_store(
            transport_plan[0], stacked_features_query, stacked_features_corpus, reverse_transport_plan=transport_plan[1]
        )[padded_edge_indices, :]

        return interaction_features, transport_plan[0]

    def propagation_step_with_post_interaction(
        self, prop_idx, from_idx, to_idx, paired_edge_counts,
        node_features_enc, edge_features_enc,
        features_to_transport_plan, padded_edge_indices
    ):
        if prop_idx != 0:
            interaction_features, transport_plan = self.end_to_end_interaction_alignment(
                edge_features_enc, paired_edge_counts, features_to_transport_plan, padded_edge_indices
            )
        else:
            interaction_features = torch.zeros_like(edge_features_enc)
            transport_plan = None

        aggregated_messages = self.prop_layer._compute_aggregated_messages(
            node_features_enc, from_idx, to_idx, edge_features_enc
        )
        node_features_enc = self.prop_layer._compute_node_update(
            node_features_enc, [aggregated_messages]
        )

        edge_features_enc = model_utils.propagation_messages(
            propagation_layer=self.prop_layer,
            node_features=node_features_enc,
            edge_features=edge_features_enc,
            from_idx=from_idx,
            to_idx=to_idx
        )

        edge_features_enc = self.interaction_encoder(torch.cat([edge_features_enc, interaction_features], dim=-1))

        return node_features_enc, edge_features_enc, transport_plan

    def propagation_step_without_interaction(
        self, prop_idx, from_idx, to_idx, paired_edge_counts,
        node_features_enc, edge_features_enc,
        features_to_transport_plan, padded_edge_indices
    ):
        aggregated_messages = self.prop_layer._compute_aggregated_messages(
            node_features_enc, from_idx, to_idx, edge_features_enc
        )
        node_features_enc = self.prop_layer._compute_node_update(
            node_features_enc, [aggregated_messages]
        )

        edge_features_enc = model_utils.propagation_messages(
            propagation_layer=self.prop_layer,
            node_features=node_features_enc,
            edge_features=edge_features_enc,
            from_idx=from_idx,
            to_idx=to_idx
        )

        return node_features_enc, edge_features_enc, None

    def neural_scoring(self, edge_features_enc, edge_graph_idx, paired_edge_counts):
        graph_vectors = self.aggregator(edge_features_enc, edge_graph_idx, 2 * len(paired_edge_counts))
        graph_vector_dim = graph_vectors.shape[-1]
        reshaped_graph_vectors = graph_vectors.reshape(-1, graph_vector_dim * 2)
        return self.scoring_mlp(reshaped_graph_vectors)[:, 0], []

    def ntn_scoring(self, edge_features_enc, edge_graph_idx, paired_edge_counts):
        graph_vectors = self.aggregator(edge_features_enc, edge_graph_idx, 2 * len(paired_edge_counts))
        graph_vector_dim = graph_vectors.shape[-1]
        reshaped_graph_vectors = graph_vectors.reshape(-1, graph_vector_dim * 2)
        query_graph_vectors = reshaped_graph_vectors[:, :graph_vector_dim]
        corpus_graph_vectors = reshaped_graph_vectors[:, graph_vector_dim:]
        return self.ntn_layer(query_graph_vectors, corpus_graph_vectors), []

    def aggregated_scoring(self, edge_features_enc, edge_graph_idx, paired_edge_counts):
        graph_vectors = self.aggregator(edge_features_enc, edge_graph_idx, 2 * len(paired_edge_counts))
        graph_vector_dim = graph_vectors.shape[-1]
        reshaped_graph_vectors = graph_vectors.reshape(-1, graph_vector_dim * 2)
        query_graph_vectors = reshaped_graph_vectors[:, :graph_vector_dim]
        corpus_graph_vectors = reshaped_graph_vectors[:, graph_vector_dim:]

        return -torch.sum(
            torch.nn.functional.relu(query_graph_vectors - corpus_graph_vectors),
            dim=-1
        ), []

    def set_aligned_scoring(self, edge_features_enc, paired_edge_counts, features_to_transport_plan):
        stacked_features_query, stacked_features_corpus = model_utils.split_and_stack(
            edge_features_enc, paired_edge_counts, self.max_edge_set_size
        )
        transport_plan = features_to_transport_plan(
            stacked_features_query, stacked_features_corpus,
            preprocessor = self.scoring_alignment_preprocessor,
            alignment_function = self.scoring_alignment_function,
            what_for = 'scoring'
        )
    
        return model_utils.feature_alignment_score(
            stacked_features_query, stacked_features_corpus, transport_plan[0]
        ), transport_plan[0]

    def forward_with_alignment(self, graphs, graph_sizes, graph_adj_matrices):
        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        paired_edge_counts = model_utils.get_paired_edge_counts(
            from_idx, to_idx, graph_idx, 2 * len(graph_sizes)
        )
        num_edges_query, num_edges_corpus = zip(*paired_edge_counts)
        num_edges_query = torch.tensor(num_edges_query, device=self.device)
        num_edges_corpus = torch.tensor(num_edges_corpus, device=self.device)
        padded_edge_indices = model_utils.get_padded_indices(paired_edge_counts, self.max_edge_set_size, self.device)

        edge_graph_idx = torch.arange(
            2 * len(graph_sizes), device=self.device
        ).repeat_interleave(self.max_edge_set_size)[padded_edge_indices]

        features_to_transport_plan = functools.partial(
            model_utils.features_to_transport_plan,
            query_sizes=num_edges_query, corpus_sizes=num_edges_corpus,
            graph_size_to_mask_map=self.graph_size_to_mask_map
        )

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)

        transport_plans = []
        for prop_idx in range(self.propagation_steps):
            node_features_enc, edge_features_enc, transport_plan = self.propagation_function(
                prop_idx, from_idx, to_idx, paired_edge_counts, node_features_enc,
                edge_features_enc, features_to_transport_plan, padded_edge_indices
            )
            if transport_plan is not None:
                transport_plans.append(transport_plan)

        ############################## SCORING ##############################
        if self.scoring == AGGREGATED:
            return self.aggregated_scoring(edge_features_enc, edge_graph_idx, paired_edge_counts)
        elif self.scoring == SET_ALIGNED:
            score, transport_plan = self.set_aligned_scoring(edge_features_enc, paired_edge_counts, features_to_transport_plan)
            transport_plans.append(transport_plan)
            return (score, transport_plan)#, torch.stack(transport_plans, dim=1)
        elif self.scoring == NEURAL:
            return self.neural_scoring(edge_features_enc, edge_graph_idx, paired_edge_counts)
        elif self.scoring == NTN:
            return self.ntn_scoring(edge_features_enc, edge_graph_idx, paired_edge_counts)
