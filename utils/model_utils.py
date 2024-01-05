import torch
from GMN.segment import unsorted_segment_sum
import torch.nn.functional as F

def get_graph_features(graphs):
    return graphs.node_features, graphs.edge_features, graphs.from_idx, graphs.to_idx, graphs.graph_idx    

def pytorch_sample_gumbel(shape, device, eps=1e-20):
    # Sample from Gumbel(0, 1)
    U = torch.rand(shape, device=device, dtype=torch.float)
    return -torch.log(eps - torch.log(U + eps))

def pytorch_sinkhorn_iters(log_alpha, device, temperature=0.1, noise_factor=1.0, num_iters=20):
    batch_size, num_objs, _ = log_alpha.shape
    noise = pytorch_sample_gumbel([batch_size, num_objs, num_objs], device) * noise_factor
    log_alpha = log_alpha + noise
    log_alpha = torch.div(log_alpha, temperature)
    for _ in range(num_iters):
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, num_objs, 1)
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, num_objs)
    return torch.exp(log_alpha)

def graph_size_to_mask_map(max_set_size, lateral_dim, device=None):
    return [torch.cat((
        torch.tensor([1], device=device, dtype=torch.float).repeat(x, 1).repeat(1, lateral_dim),
        torch.tensor([0], device=device, dtype=torch.float).repeat(max_set_size - x, 1).repeat(1, lateral_dim)
    )) for x in range(0, max_set_size + 1)]

def split_to_query_and_corpus(features, graph_sizes):
    # [(8, 12), (10, 13), (10, 14)] -> [8, 12, 10, 13, 10, 14]
    graph_sizes_flat  = [item for sublist in graph_sizes for item in sublist]
    features_split = torch.split(features, graph_sizes_flat, dim=0)
    features_query = features_split[0::2]
    features_corpus = features_split[1::2]
    return features_query, features_corpus

def split_and_stack(features, graph_sizes, max_set_size):
    features_query, features_corpus = split_to_query_and_corpus(features, graph_sizes)
    
    stack_features = lambda features_array: torch.stack([
        F.pad(features, pad=(0, 0, 0, max_set_size - features.shape[0])) for features in features_array
    ])
    return stack_features(features_query), stack_features(features_corpus)

def feature_alignment_score(query_features, corpus_features, transport_plan):
    return - torch.maximum(
        query_features - transport_plan @ corpus_features,
        torch.tensor([0], device=query_features.device)
    ).sum(dim=(1, 2))

def get_paired_edge_counts(from_idx, to_idx, graph_idx, num_graphs):
    edges_per_src_node = unsorted_segment_sum(torch.ones_like(from_idx, dtype=torch.float), from_idx, len(graph_idx))
    edges_per_graph_from = unsorted_segment_sum(edges_per_src_node, graph_idx, num_graphs)

    edges_per_dest_node = unsorted_segment_sum(torch.ones_like(to_idx, dtype=torch.float), to_idx, len(graph_idx))
    edges_per_graph_to = unsorted_segment_sum(edges_per_dest_node, graph_idx, num_graphs)

    assert (edges_per_graph_from == edges_per_graph_to).all()

    return edges_per_graph_to.reshape(-1, 2).int().tolist()

def propagation_messages(propagation_layer, node_features, edge_features, from_idx, to_idx):
    edge_src_features = node_features[from_idx]
    edge_dest_features = node_features[to_idx]

    forward_edge_msg = propagation_layer._message_net(torch.cat([
        edge_src_features, edge_dest_features, edge_features
    ], dim=-1))
    backward_edge_msg = propagation_layer._reverse_message_net(torch.cat([
        edge_dest_features, edge_src_features, edge_features
    ], dim=-1))
    return forward_edge_msg + backward_edge_msg