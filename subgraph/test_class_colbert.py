import timeit
import torch
import numpy as np
from subgraph.iso_matching_models import colbert_scores_for_gmn_data

max_node_count = 11

def colbert_scoring_by_indra(node_features_enc, batch_data_sizes, graph_idx):
    node_counts = torch.tensor(batch_data_sizes, device=node_features_enc.device)
    node_feature_enc_split = torch.split(node_features_enc, node_counts.flatten().tolist(), dim=0)
    node_feature_enc_query = node_feature_enc_split[0::2]
    node_feature_enc_corpus = node_feature_enc_split[1::2]
    stacked_qnode_emb = torch.stack([torch.nn.functional.pad(x, pad=(0,0,0,max_node_count-x.shape[0])) \
                                        for x in node_feature_enc_query])
    stacked_cnode_emb = torch.stack([torch.nn.functional.pad(x, pad=(0,0,0,max_node_count-x.shape[0])) \
                                        for x in node_feature_enc_corpus])
    scores = (stacked_qnode_emb[:,:,None,:]*stacked_cnode_emb[:,None,:,:]).sum(-1)
    
    def get_padding_mask(node_counts):
        return torch.arange(0, max_node_count, device=node_counts.device)[None,:] < node_counts[:,None]

    combined_padding_mask = get_padding_mask(node_counts[:, 0])[:,:,None] * ~get_padding_mask(node_counts[:, 1])[:,None,:]
    scores.masked_fill_(combined_padding_mask, -torch.inf)
    return scores.max(-1).values.sum(-1)

node_features_enc = torch.randn(10, 10).cuda()
batch_data_sizes = [(2, 3), (1, 4)]
graph_idx = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 3, 3]).cuda()

print("Our implementation")
# print(colbert_scores_for_gmn_data(node_features_enc, batch_data_sizes, graph_idx))
print(timeit.timeit(lambda: colbert_scores_for_gmn_data(node_features_enc, batch_data_sizes, graph_idx), number=100))

print("Indra's implementation")
# print(colbert_scoring_by_indra(node_features_enc, batch_data_sizes, graph_idx))
print(timeit.timeit(lambda: colbert_scoring_by_indra(node_features_enc, batch_data_sizes, graph_idx), number=100))

for idx in range(1000):
    node_features_enc = torch.randn(10, 200).cuda()
    our_score = colbert_scores_for_gmn_data(node_features_enc, batch_data_sizes, graph_idx)
    indra_score = colbert_scoring_by_indra(node_features_enc, batch_data_sizes, graph_idx)
    if torch.sum((our_score - indra_score) ** 2) > 1e-5:
        print(node_features_enc)
        print("Mismatch", idx, ":", our_score, indra_score)