import torch
import torch.nn.functional as F
from utils import model_utils
from utils.tooling import ReadOnlyConfig
import GMN.graphembeddingnetwork as gmngen

VALID_ATTENTION_MODES = {
    'q_to_c': lambda x, y: x,
    'c_to_q': lambda x, y: y,
    'min': torch.minimum,
    'max': torch.maximum,
}

class NodeAlignNodeLossAttention(torch.nn.Module):
    def __init__(
        self,
        max_node_set_size,
        max_edge_set_size,
        encoder_config: ReadOnlyConfig,
        propagation_layer_config: ReadOnlyConfig,
        propagation_steps,
        attention_config: ReadOnlyConfig,
        attention_feature_dim,
        attention_mode,
        masked_attention,
        device
    ):
        super(NodeAlignNodeLossAttention, self).__init__()
        self.max_node_set_size = max_node_set_size
        self.attention_mode = attention_mode
        self.attention_config = attention_config
        self.attention_func = model_utils.masked_attention if masked_attention else (
            lambda log_alpha, query_sizes, corpus_sizes, temperature:
            model_utils.attention(log_alpha=log_alpha, temperature=temperature)
        ) 
        assert attention_mode in VALID_ATTENTION_MODES, f"attention_mode should be one of {list(VALID_ATTENTION_MODES.keys())}"
        self.device = device

        self.graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
            max_set_size=max_node_set_size, lateral_dim=attention_feature_dim, device=self.device
        )

        self.encoder = gmngen.GraphEncoder(**encoder_config)
        self.prop_layer = gmngen.GraphPropLayer(**propagation_layer_config)
        self.propagation_steps = propagation_steps

        self.attention_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(propagation_layer_config.node_state_dim, attention_feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(attention_feature_dim, attention_feature_dim)
        )

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
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
        transformed_features_query = self.attention_feature_layers(stacked_features_query)
        transformed_features_corpus = self.attention_feature_layers(stacked_features_corpus)

        def mask_graphs(features, graph_sizes):
            mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
            return mask * features
        masked_features_query = mask_graphs(transformed_features_query, query_sizes)
        masked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

        attention_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
        q_to_c_matrix, c_to_q_matrix = self.attention_func(
            log_alpha=attention_input,
            query_sizes=query_sizes,
            corpus_sizes=corpus_sizes,
            **self.attention_config
        )

        q_to_c_score = model_utils.feature_alignment_score(stacked_features_query, stacked_features_corpus, q_to_c_matrix)
        c_to_q_score = model_utils.reversed_feature_alignment_score(stacked_features_query, stacked_features_corpus, c_to_q_matrix)
        return VALID_ATTENTION_MODES[self.attention_mode](q_to_c_score, c_to_q_score)
