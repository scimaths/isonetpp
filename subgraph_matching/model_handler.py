from utils.tooling import read_config
import subgraph_matching.dataset as dataset

from subgraph_matching.models.node_align_node_loss import NodeAlignNodeLoss
from subgraph_matching.models.nanl_consistency import NodeAlignNodeLossConsistency
from subgraph_matching.models.isonet import ISONET
from subgraph_matching.models.node_early_interaction import NodeEarlyInteraction
from subgraph_matching.models.edge_early_interaction import EdgeEarlyInteraction
from subgraph_matching.models.nanl_attention import NodeAlignNodeLossAttention

model_name_to_class_mappings = {
    'node_align_node_loss': NodeAlignNodeLoss,
    'nanl_consistency': NodeAlignNodeLossConsistency,
    'isonet': ISONET,
    'node_early_interaction': NodeEarlyInteraction,
    'edge_early_interaction': EdgeEarlyInteraction,
    'nanl_attention_q_to_c': NodeAlignNodeLossAttention,
    'nanl_attention_c_to_q': NodeAlignNodeLossAttention,
    'nanl_attention_max': NodeAlignNodeLossAttention,
    'nanl_attention_min': NodeAlignNodeLossAttention,
    'nanl_masked_attention_q_to_c': NodeAlignNodeLossAttention,
    'nanl_masked_attention_c_to_q': NodeAlignNodeLossAttention,
    'nanl_masked_attention_max': NodeAlignNodeLossAttention,
    'nanl_masked_attention_min': NodeAlignNodeLossAttention,
}

def get_model_names():
    return list(model_name_to_class_mappings.keys())

def get_model(model_name, config_path, max_node_set_size, max_edge_set_size, device):
    assert model_name in get_model_names(), f"Model {model_name} not defined; choose one from {', '.join(get_model_names())}"

    config = read_config(config_path)
    assert config.name == model_name, f"Model names in args and config differ - {model_name} v/s {config.name}"

    return model_name_to_class_mappings[model_name](
        max_node_set_size=max_node_set_size,
        max_edge_set_size=max_edge_set_size,
        device=device,
        **config.model_config
    )

def get_data_type_for_model(model_name):
    return dataset.GMN_DATA_TYPE