import torch
from typing import Optional
from utils import model_utils
from utils.tooling import ReadOnlyConfig
import GMN.graphembeddingnetwork as gmngen
from subgraph_matching.models._template import AlignmentModel

# Alignment preprocessing constants
LRL = 'lrl'
IDENTITY = 'identity'
POSSIBLE_ALIGNMENT_PREPROCESSORS = [LRL, IDENTITY]

# Alignment constants
ATTENTION = 'attention'
SINKHORN = 'sinkhorn'
POSSIBLE_ALIGNMENTS = [ATTENTION, SINKHORN, None]

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
        scoring_alignment_preprocessor: str = IDENTITY, # one of 'lrl' or 'identity'
        scoring_alignment_feature_dim: Optional[int] = None,
        # Use scoring arguments for interaction
        unify_scoring_and_interaction: bool = False,
        # Arguments to manage interaction-time alignment
        interaction_alignment: Optional[str] = ATTENTION, # one of 'attention' or 'sinkhorn' if not unified, else None
        interaction_alignment_preprocessor: str = IDENTITY, # one of 'lrl' or 'identity'
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
        assert (scoring_alignment_preprocessor == LRL) ^ (scoring_alignment_feature_dim is None), (
            "`scoring_alignment_feature_dim` should be non-zero iff LRL preprocessing is used"
        )
        # ensure no extra params if aggregated
        assert (scoring != AGGREGATED) or (scoring_alignment_preprocessor == IDENTITY), (
            "aggregated scoring must have identity preprocessor to prevent extra parameters"
        )
        self.scoring = scoring
        self.aggregator_config = aggregator_config
        self.scoring_alignment = scoring_alignment
        self.scoring_alignment_feature_dim = scoring_alignment_feature_dim
        self.scoring_alignment_preprocessor = scoring_alignment_preprocessor

        #########################################
        # CONSTRAINTS for interaction
        assert not(unify_scoring_and_interaction) or (scoring != AGGREGATED), (
            "Can't unify with aggregated scoring"
        )
        assert unify_scoring_and_interaction ^ (interaction_alignment in POSSIBLE_ALIGNMENTS and interaction_alignment is not None), (
            f"`interaction_alignment` must be one of {POSSIBLE_ALIGNMENTS} and not None iff not-unified else None, found {interaction_alignment}"
        )
        # require feature_dim for LRL preprocessing
        assert (interaction_alignment_preprocessor == LRL) ^ (interaction_alignment_feature_dim is None), (
            "`interaction_alignment_feature_dim` should be non-zero iff LRL preprocessing is used"
        )
        self.unify_scoring_and_interaction = unify_scoring_and_interaction
        self.interaction_alignment = interaction_alignment
        self.interaction_alignment_feature_dim = interaction_alignment_feature_dim
        self.interaction_alignment_preprocessor = interaction_alignment_preprocessor

        #########################################
        # CONSTRAINTS for configs
        assert (sinkhorn_config is None) ^ SINKHORN in [scoring_alignment, interaction_alignment], (
            "`sinkhorn_config` was specified but it was not used in scoring or interaction or vice-versa"
        )
        assert (attention_config is None) ^ ATTENTION in [scoring_alignment, interaction_alignment], (
            "`attention_config` was specified but it was not used in scoring or interaction or vice-versa"
        )

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
        self.setup_scoring()
        self.setup_interaction()


        ####################################################################
        #  NOT GONNA USE BELOW CODE, TOO COMPLEX DICTIONARY MANIPULATIONS  #
        ####################################################################

        # Handle unification for LRL featurization
        # alignment_to_config_dict = {
        #     SINKHORN: sinkhorn_config,
        #     ATTENTION: attention_config,
        #     None: None
        # }
        # self.alignment_config = {'scoring': alignment_to_config_dict[scoring_alignment]}
        # self.alignment_config['interaction'] = self.alignment_config['scoring'] if unify_scoring_and_interaction else (
        #     alignment_to_config_dict[interaction_alignment]
        # )

        # self.alignment_featurization = {}
        # create_featurizer = lambda preprocessor, feature_dim: {
        #     IDENTITY: lambda x: x,
        #     LRL: lambda x: torch.nn.Sequential(
        #         torch.nn.Linear(propagation_layer_config.node_state_dim, feature_dim),
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(feature_dim, feature_dim)
        #     )(x)
        # }[preprocessor]
        # self.alignment_featurization['scoring'] = create_featurizer(scoring_alignment_preprocessor, scoring_alignment_feature_dim)
        # self.alignment_featurization['interaction'] = self.alignment_featurization['scoring'] if unify_scoring_and_interaction else (
        #     create_featurizer(interaction_alignment_preprocessor, interaction_alignment_feature_dim)
        # )

    def setup_scoring(self):
        if self.scoring == AGGREGATED:
            self.aggregator = gmngen.GraphAggregator(**self.aggregator_config)
        elif self.scoring == SET_ALIGNED:
            pass
        else:
            raise NotImplementedError(f"scoring is implemented only for these modes - {POSSIBLE_SCORINGS}")

    def perform_scoring(self):
        pass

    def setup_interaction(self):
        if self.unify_scoring_and_interaction:
            pass
        else:
            pass

    def perform_interaction(self):
        pass

    def forward_with_alignment(self, graphs, graph_sizes, graph_adj_matrices):
        query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = torch.tensor(query_sizes, device=self.device)
        corpus_sizes = torch.tensor(corpus_sizes, device=self.device)

        node_features, edge_features, from_idx, to_idx, _ = model_utils.get_graph_features(graphs)

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for _ in range(self.propagation_steps) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, edge_features_enc)
        
        stacked_features_query, stacked_features_corpus = model_utils.split_and_stack(
            node_features_enc, graph_sizes, self.max_node_set_size
        )
        transformed_features_query = self.sinkhorn_feature_layers(stacked_features_query)
        transformed_features_corpus = self.sinkhorn_feature_layers(stacked_features_corpus)

        def mask_graphs(features, graph_sizes):
            mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
            return mask * features
        masked_features_query = mask_graphs(transformed_features_query, query_sizes)
        masked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

        sinkhorn_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
        transport_plan = model_utils.pytorch_sinkhorn_iters(log_alpha=sinkhorn_input, device=self.device, **self.sinkhorn_config)
        
        return model_utils.feature_alignment_score(stacked_features_query, stacked_features_corpus, transport_plan), [transport_plan, transport_plan.transpose(-1, -2)]
