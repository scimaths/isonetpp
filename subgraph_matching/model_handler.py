import subgraph_matching.dataset as dataset

from subgraph_matching.models.node_align_node_loss import NodeAlignNodeLoss
from subgraph_matching.models.nanl_consistency import NodeAlignNodeLossConsistency
from subgraph_matching.models.isonet import ISONET
from subgraph_matching.models.node_early_interaction import NodeEarlyInteraction
from subgraph_matching.models.node_early_interaction_2 import NodeEarlyInteraction2
from subgraph_matching.models.node_early_interaction_3 import NodeEarlyInteraction3
from subgraph_matching.models.edge_early_interaction_1 import EdgeEarlyInteraction1
from subgraph_matching.models.edge_early_interaction_baseline_1 import EdgeEarlyInteractionBaseline1
from subgraph_matching.models.edge_early_interaction_2 import EdgeEarlyInteraction2
from subgraph_matching.models.edge_early_interaction_3 import EdgeEarlyInteraction3
from subgraph_matching.models.node_edge_early_interaction import NodeEdgeEarlyInteraction
from subgraph_matching.models.edge_early_interaction import EdgeEarlyInteraction
from subgraph_matching.models.nanl_attention import NodeAlignNodeLossAttention
from subgraph_matching.models.gmn_baseline import GMNBaseline
from subgraph_matching.models.gmn_iterative_refinement import GMNIterativeRefinement
from subgraph_matching.models.graphsim import GraphSim
from subgraph_matching.models.egsc_modified import EGSC as EGSC_Modified
from subgraph_matching.models.eric import ERIC
from subgraph_matching.models.gotsim import GOTSim
from subgraph_matching.models.gmn_embed import GMN_embed_hinge
from subgraph_matching.models.h2mn import H2MN
from subgraph_matching.models.greed import Greed
from subgraph_matching.models.neuromatch import NeuroMatch
from subgraph_matching.models.simgnn import SimGNN

model_name_to_class_mappings = {
    'node_align_node_loss': NodeAlignNodeLoss,
    'nanl_consistency': NodeAlignNodeLossConsistency,
    'isonet': ISONET,
    'node_early_interaction': NodeEarlyInteraction,
    'node_early_interaction_2': NodeEarlyInteraction2,
    'node_early_interaction_3': NodeEarlyInteraction3,
    'node_early_interaction_consistency': NodeEarlyInteraction,
    'node_edge_early_interaction': NodeEdgeEarlyInteraction,
    'node_edge_early_interaction_consistency': NodeEdgeEarlyInteraction,
    'edge_early_interaction': EdgeEarlyInteraction,
    'edge_early_interaction_1': EdgeEarlyInteraction1,
    'edge_early_interaction_1_baseline': EdgeEarlyInteractionBaseline1,
    'edge_early_interaction_2': EdgeEarlyInteraction2,
    'edge_early_interaction_3': EdgeEarlyInteraction3,
    'edge_early_interaction_consistency': EdgeEarlyInteraction,
    'nanl_attention_q_to_c': NodeAlignNodeLossAttention,
    'nanl_attention_c_to_q': NodeAlignNodeLossAttention,
    'nanl_attention_max': NodeAlignNodeLossAttention,
    'nanl_attention_min': NodeAlignNodeLossAttention,
    'nanl_masked_attention_q_to_c': NodeAlignNodeLossAttention,
    'nanl_masked_attention_c_to_q': NodeAlignNodeLossAttention,
    'nanl_masked_attention_max': NodeAlignNodeLossAttention,
    'nanl_masked_attention_min': NodeAlignNodeLossAttention,
    'graphsim': GraphSim,
    'egsc_modified': EGSC_Modified,
    'eric': ERIC,
    'gotsim': GOTSim,
    'gmn_embed': GMN_embed_hinge,
    'H2MN': H2MN,
    'neuromatch': NeuroMatch,
    'greed': Greed,
    'simgnn': SimGNN
}

def get_model_names():
    return list(model_name_to_class_mappings.keys())

def get_model(model_name, config, max_node_set_size, max_edge_set_size, device):
    if model_name.startswith('gmn_baseline'):
        model_class = GMNBaseline
    elif model_name.startswith('gmn_iterative_refinement'):
        model_class = GMNIterativeRefinement
    elif model_name.startswith('edge_early_interaction_baseline_1'):
        model_class = EdgeEarlyInteractionBaseline1
    elif model_name.startswith('edge_early_interaction_1'):
        model_class = EdgeEarlyInteraction1
    elif model_name.startswith('isonet'):
        model_class = ISONET
    else:
        model_class = model_name_to_class_mappings[model_name]

    return model_class(
        max_node_set_size=max_node_set_size,
        max_edge_set_size=max_edge_set_size,
        device=device,
        **config
    )

def get_data_type_for_model(model_name):
    if model_name in ['graphsim', 'egsc', 'egsc_modified', 'eric', 'gotsim', 'H2MN', 'greed', 'neuromatch']:
        return dataset.PYG_DATA_TYPE
    return dataset.GMN_DATA_TYPE
