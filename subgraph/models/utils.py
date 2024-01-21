import torch
from subgraph.utils import cudavar

def pytorch_sample_gumbel(av,shape, eps=1e-20):
  #Sample from Gumbel(0, 1)
  device = 'cuda:0' if av.has_cuda and av.want_cuda else 'cpu'
  U = torch.rand(shape, dtype=torch.float32, device=device)
  return -torch.log(eps - torch.log(U + eps))

def pytorch_sinkhorn_iters(av, log_alpha,temp=0.1,noise_factor=1.0, n_iters = 20):
    noise_factor = av.NOISE_FACTOR
    batch_size = log_alpha.size()[0]
    n = log_alpha.size()[1]
    log_alpha = log_alpha.view(-1, n, n)
    noise = pytorch_sample_gumbel(av,[batch_size, n, n])*noise_factor
    log_alpha = log_alpha + noise
    log_alpha = torch.div(log_alpha,temp)
    for i in range(n_iters):
      log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)
      log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
    return torch.exp(log_alpha)

def colbert_scores_for_gmn_data(node_features_enc, batch_data_sizes, graph_idx):  
    num_nodes, enc_dim = node_features_enc.shape
    num_pairs = len(batch_data_sizes)

    def get_true_indices_on_cuda(bool_tensor):
        return bool_tensor.argsort()[num_nodes - int(bool_tensor.sum()):].type(torch.LongTensor).cuda()

    corpus_bool = graph_idx % 2
    corpus_indices = get_true_indices_on_cuda(corpus_bool)
    # SUM(NODE_COUNT(CORPUS)) X ENC_DIM
    corpus_features_enc = node_features_enc.gather(dim=0, index=corpus_indices.unsqueeze(1) + torch.zeros(1, enc_dim, dtype=int).cuda())
    corpus_scatter_index = ((graph_idx[corpus_indices] - 1)/2).type(torch.LongTensor).cuda()

    query_bool = 1 - corpus_bool
    query_indices = get_true_indices_on_cuda(query_bool)
    # SUM(NODE_COUNT(QUERY)) X ENC_DIM
    query_features_enc = node_features_enc.gather(dim=0, index=query_indices.unsqueeze(1) + torch.zeros(1, enc_dim, dtype=int).cuda())
    query_scatter_index = (graph_idx[query_indices]/2).type(torch.LongTensor).cuda()

    node_similarities = query_features_enc @ corpus_features_enc.T
    # UNCOMMENT TO FIND NEGATIVE SIMILARITIES BETWEEN QUERY AND CORPUS NODES
    # negative_similarity_count = (node_similarities < 0).sum()
    # if negative_similarity_count > 0:
    #   print("Negative found", node_similarities, node_similarities[node_similarities < 0])
    non_correspondence_mask = (query_scatter_index.unsqueeze(1) - corpus_scatter_index.unsqueeze(0)) != 0
    node_similarities.masked_fill_(non_correspondence_mask, -torch.inf)
    max_nodewise_similarity = node_similarities.max(dim=1).values
    scores = torch.zeros(num_pairs, dtype=node_similarities.dtype).cuda().scatter_add_(0, query_scatter_index, max_nodewise_similarity)

    return scores