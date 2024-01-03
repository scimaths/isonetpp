import torch
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