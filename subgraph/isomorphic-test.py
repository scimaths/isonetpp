import os
import torch
import pickle
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
from subgraph.utils import cudavar
from subgraph import iso_matching_models as im

def fetch_edge_counts(to_idx,from_idx,graph_idx,num_graphs):
    from GMN.segment import unsorted_segment_sum
    tt = unsorted_segment_sum(torch.ones(len(to_idx)), to_idx, len(graph_idx))
    tt1 = unsorted_segment_sum(torch.ones(len(from_idx)), from_idx, len(graph_idx))
    edge_counts = unsorted_segment_sum(tt, graph_idx, num_graphs)
    edge_counts1 = unsorted_segment_sum(tt1, graph_idx, num_graphs)
    assert(edge_counts == edge_counts1).all()
    assert(sum(edge_counts)== len(to_idx))
    return list(map(int,edge_counts.tolist()))

def get_graph( batch):
    graph = batch
    from_idx = torch.from_numpy(graph.from_idx).long()
    to_idx = torch.from_numpy(graph.to_idx).long()
    graph_idx = torch.from_numpy(graph.graph_idx).long()
    return from_idx, to_idx, graph_idx

def isomorphic(av,sampler1,sampler2):

    # Q1 = nx.Graph()
    # Q2 = nx.Graph()
    # edges1 = [(0,1),(1,2),(2,0),(4,0)]
    # edges2 = [(1,2),(0,2),(1,0)]
    # Q1.add_edges_from(edges1)
    # Q2.add_edges_from(edges2)
    # print(nx.is_isomorphic(Q1, Q2))
    # exit()

    d_pos = sampler1.list_pos
    d_neg = sampler1.list_neg
    q_graphs1 = list(range(len(sampler1.query_graphs)))
    q_graphs2 = list(range(len(sampler2.query_graphs)))
    
    q1 = []
    for q_id in tqdm(q_graphs1):
        dpos = list(filter(lambda x:x[0][0]==q_id,d_pos))
        dneg = list(filter(lambda x:x[0][0]==q_id,d_neg))
        d = dpos+dneg
        n_batches = sampler1.create_batches(d)
        batch_data,batch_data_sizes,_,batch_adj = sampler1.fetch_batched_data_by_id(0)
        from_idx, to_idx, graph_idx = get_graph(batch_data)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]
        edge_counts  = fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
        from_idx_suff = torch.cat([torch.tensor([sum(batch_data_sizes_flat[:node_idx])]).repeat(edge_counts[node_idx]) for node_idx in range(len(edge_counts))])
        from_idx -= from_idx_suff
        to_idx -= from_idx_suff

        query_from = from_idx[0:edge_counts[0]].tolist()
        query_to = to_idx[0:edge_counts[0]].tolist()
        query_edges = [(query_from[idx], query_to[idx]) for idx in range(len(query_from))]
        Query = nx.Graph()
        Query.add_edges_from(query_edges)
        q1.append(Query)
    
    q2 = []
    for q_id in tqdm(q_graphs2):
        dpos = list(filter(lambda x:x[0][0]==q_id,d_pos))
        dneg = list(filter(lambda x:x[0][0]==q_id,d_neg))
        d = dpos+dneg
        n_batches = sampler2.create_batches(d)
        batch_data,batch_data_sizes,_,batch_adj = sampler2.fetch_batched_data_by_id(0)
        from_idx, to_idx, graph_idx = get_graph(batch_data)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]
        edge_counts  = fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
        from_idx_suff = torch.cat([torch.tensor([sum(batch_data_sizes_flat[:node_idx])]).repeat(edge_counts[node_idx]) for node_idx in range(len(edge_counts))])
        from_idx -= from_idx_suff
        to_idx -= from_idx_suff

        query_from = from_idx[0:edge_counts[0]].tolist()
        query_to = to_idx[0:edge_counts[0]].tolist()
        query_edges = [(query_from[idx], query_to[idx]) for idx in range(len(query_from))]
        Query = nx.Graph()
        Query.add_edges_from(query_edges)
        q2.append(Query)

    for qa in q1:
        for qb in q2:
            if nx.is_isomorphic(qa, qb):
                print(qa,qb)

def fetch_gmn_data(av):
    train_data = im.OurMatchingModelSubgraphIsoData(av,mode="train")
    test_data = im.OurMatchingModelSubgraphIsoData(av,mode="test")
    train_data.data_type = "gmn"
    test_data.data_type = "gmn"
    return train_data, test_data

def get_result(av):
    train_data, test_data = fetch_gmn_data(av)
    isomorphic(av, train_data, test_data)

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", type=str)
ad = ap.parse_args()
test_model_dir = ad.model_dir

for model_loc in sorted(os.listdir(test_model_dir)):
    saved = torch.load(os.path.join(test_model_dir, model_loc))
    av = saved['av']
    av.prop_separate_params = False
    av.want_cuda = True
    get_result(av)
    break