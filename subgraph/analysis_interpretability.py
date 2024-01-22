import os
import torch
import pickle
import argparse
import numpy as np
import networkx as nx
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from subgraph.utils import cudavar
from GMN.configure import get_default_config
import networkx.algorithms.isomorphism as iso
from subgraph import iso_matching_models as im
from subgraph.earlystopping import EarlyStoppingModule
from sklearn.metrics import average_precision_score, ndcg_score

def load_config(av):
  config = get_default_config()

  config['seed'] = av.SEED

  config['encoder'] ['node_hidden_sizes'] = [10]
  config['encoder'] ['node_feature_dim'] = 1
  config['encoder'] ['edge_feature_dim'] = 1
    
  config['aggregator'] ['node_hidden_sizes'] = [10]
  config['aggregator'] ['graph_transform_sizes'] = [10]
  config['aggregator'] ['input_size'] = [10]

  config['graph_matching_net'] ['node_state_dim'] = 10
  config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_matching_net'] ['edge_hidden_sizes'] = [20]
  config['graph_matching_net'] ['node_hidden_sizes'] = [10]
    
  config['graph_embedding_net'] ['node_state_dim'] = 10
  config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_embedding_net'] ['edge_hidden_sizes'] = [20]
  config['graph_embedding_net'] ['node_hidden_sizes'] = [10]
  config['fringe_isonet'] ['masking_for_msg_passing_count'] = av.MASKING_FOR_MSG_PASSING_COUNT

  config['early_interaction'] = {
    'n_time_updates': av.time_updates,
    'time_update_idx': av.time_update_idx,
    'prop_separate_params': av.prop_separate_params
  }

  config['node_early_interaction_interpretability'] = {
    'lambd' : av.lambd
  }
  
  config['graphsim']= {}
  config['graphsim']['conv_kernel_size'] = [10,4,2]
  config['graphsim']['linear_size'] = [24, 16]
  config['graphsim']['gcn_size'] = [10,10,10]
  config['graphsim']['conv_pool_size'] = [3,3,2]
  config['graphsim']['conv_out_channels'] = [2,4,8]
  config['graphsim']['dropout'] = av.dropout

  config['training']['batch_size']  = av.BATCH_SIZE
  config['training']['margin']  = av.MARGIN
  config['evaluation']['batch_size']  = av.BATCH_SIZE
  config['model_type']  = "embedding"
    
  return config

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

def get_alignment_edges(mapping, query_edges, corpus_edges, device):

  num_query_edges = len(query_edges)
  num_corpus_edges = len(corpus_edges)

  edges_corpus_to_idx = {corpus_edges[idx]: idx for idx in range(len(corpus_edges))}
  reverse_mapping = {mapping[key]: key for key in mapping}

  s_hat = torch.zeros((num_query_edges, num_corpus_edges), device=device)

  for edge_idx in range(num_query_edges):

    corpus_edge_1 = (reverse_mapping[query_edges[edge_idx][0]], reverse_mapping[query_edges[edge_idx][1]])
    corpus_edge_2 = (reverse_mapping[query_edges[edge_idx][1]], reverse_mapping[query_edges[edge_idx][0]])

    if corpus_edge_1 in edges_corpus_to_idx:
      s_hat[edge_idx][edges_corpus_to_idx[corpus_edge_1]] = 1

    if corpus_edge_2 in edges_corpus_to_idx:
      s_hat[edge_idx][edges_corpus_to_idx[corpus_edge_2]] = 1

  return s_hat

def get_norm_qc_pair_edges(query_edges, corpus_edges, transport_plans):

  Query = nx.Graph()
  Query.add_edges_from(query_edges)

  Corpus = nx.Graph()
  Corpus.add_edges_from(corpus_edges)

  GM = iso.GraphMatcher(Corpus,Query)
  norm = torch.zeros(transport_plans.shape[0])

  for mapping in GM.subgraph_isomorphisms_iter():
    s_hat = get_alignment_edges(mapping, query_edges, corpus_edges, transport_plans.device)
    norm = torch.maximum(norm, torch.sum(transport_plans[:, :len(query_edges), :len(corpus_edges)] * s_hat.unsqueeze(0), dim=(1,2)))
  return norm

def evaluate_improvement_edges(av,model,sampler,lambd=1):
  model.eval()

  d_pos = sampler.list_pos
  d_neg = sampler.list_neg
  q_graphs = list(range(len(sampler.query_graphs)))

  norms_total = []

  for q_id in tqdm(q_graphs):
    dpos = list(filter(lambda x:x[0][0]==q_id,d_pos))
    dneg = list(filter(lambda x:x[0][0]==q_id,d_neg))
    npos = len(dpos)
    nneg = len(dneg)
    d = dpos+dneg

    query_information = []

    if npos>0 and nneg>0:

      n_batches = sampler.create_batches(d)
      norms_per_query = []

      for n_batches_idx in tqdm(range(n_batches)): 

        batch_data,batch_data_sizes,_,batch_adj = sampler.fetch_batched_data_by_id(n_batches_idx)
        from_idx, to_idx, graph_idx = get_graph(batch_data)

        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]
        edge_counts  = fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
        from_idx_suff = torch.cat([torch.tensor([sum(batch_data_sizes_flat[:node_idx])]).repeat(edge_counts[node_idx]) for node_idx in range(len(edge_counts))])
        from_idx -= from_idx_suff
        to_idx -= from_idx_suff

        transport_plans = model(batch_data,batch_data_sizes,batch_adj).data.cpu()
        batch_size = transport_plans.shape[0]

        for batch_idx in range(batch_size):
          query_from = from_idx[sum(edge_counts[:2*batch_idx]):sum(edge_counts[:2*batch_idx+1])].tolist()
          query_to = to_idx[sum(edge_counts[:2*batch_idx]):sum(edge_counts[:2*batch_idx+1])].tolist()

          corpus_from = from_idx[sum(edge_counts[:2*batch_idx+1]):sum(edge_counts[:2*batch_idx+2])].tolist()
          corpus_to = to_idx[sum(edge_counts[:2*batch_idx+1]):sum(edge_counts[:2*batch_idx+2])].tolist()

          query_edges = [(query_from[idx], query_to[idx]) for idx in range(len(query_from))]
          corpus_edges = [(corpus_from[idx], corpus_to[idx]) for idx in range(len(corpus_from))]

          norms_per_pair = get_norm_qc_pair_edges(query_edges, corpus_edges, transport_plans[batch_idx])
          norms_per_query.append(norms_per_pair)
          query_information.append((query_edges, corpus_edges, transport_plans[batch_idx]))
    norms_total.extend(norms_per_query[:npos])
    norms_positive_pairs = torch.stack(norms_per_query[:npos])
    increase_score_idx = torch.argsort(norms_positive_pairs[:,-1])
    for idx in increase_score_idx:
      information = query_information[idx]

      Query = nx.Graph()
      Query.add_edges_from(information[0])  
      nx.draw_spring(Query, with_labels = True)
      plt.savefig(f"plots/case{idx}_query.png")
      plt.clf()

      Corpus = nx.Graph()
      Corpus.add_edges_from(information[1])
      nx.draw_spring(Corpus, with_labels = True)
      plt.savefig(f"plots/case{idx}_corpus.png")
      plt.clf()

      for trns_idx in range(3):
        curr_transport_plan = information[-1][trns_idx]
        curr_corpus_edges = information[1]
        with open(f"plots/case{idx}_mapping_{trns_idx}.txt", "w+") as f:
          for query_edge_idx in range(len(information[0])):
            corpus_edge_idx = torch.argmax(curr_transport_plan[query_edge_idx])
            f.write(str(information[0][query_edge_idx]) + " -> " + str(curr_corpus_edges[corpus_edge_idx]) + " with " + str(curr_transport_plan[query_edge_idx][corpus_edge_idx].item()) + "\n")
            curr_transport_plan = torch.cat((curr_transport_plan[:, :corpus_edge_idx], curr_transport_plan[:, corpus_edge_idx + 1:]), dim=1)
            curr_corpus_edges = curr_corpus_edges[:corpus_edge_idx] + curr_corpus_edges[corpus_edge_idx + 1:]
      input()

  return norms_total

def fetch_gmn_data(av):
    data_mode = "test" if av.test_size==25 else "Extra_test_300"
    test_data = im.OurMatchingModelSubgraphIsoData(av,mode=data_mode)
    val_data = im.OurMatchingModelSubgraphIsoData(av,mode="val")
    test_data.data_type = "pyg"
    val_data.data_type = "pyg"
    return val_data, test_data

def get_result(av,model_loc,state_dict):
    val_data, test_data = fetch_gmn_data(av)
    if model_loc.startswith("node_edge_early_interaction_with_consistency_and_two_sinkhorns"):
      config = load_config(av)
      model = im.NodeEdgeEarlyInteractionWithConsistencyAndTwoSinkhorns(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("node_edge_early_interaction_with_consistency"):
      config = load_config(av)
      model = im.NodeEdgeEarlyInteractionWithConsistency(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("node_early_interaction_interpretability"):
      config = load_config(av)
      model = im.NodeEarlyInteractionInterpretability(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("node_early_interaction"):
      config = load_config(av)
      model = im.NodeEarlyInteraction(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("edge_early_interaction"):
      config = load_config(av)
      model = im.EdgeEarlyInteraction(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    elif model_loc.startswith("isonet"):
      config = load_config(av)
      model = im.ISONET(av,config,1).to(device)
      test_data.data_type = "gmn"
      val_data.data_type = "gmn"
    else:
      print("ALERT!! CHECK FOR ERROR")  
    model.eval()
    model.load_state_dict(state_dict)
    if av.type == "edge":
      evaluate_improvement_edges(av,model,val_data)
    elif av.type == "node":
      evaluate_improvement_nodes(av,model,val_data)

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", type=str)
ap.add_argument("--type", type=str)
ad = ap.parse_args()

test_model_dir = ad.model_dir
for model_loc in sorted(os.listdir(test_model_dir)):
    saved = torch.load(os.path.join(test_model_dir, model_loc))
    device = "cuda"
    av = saved['av']
    av.test_size = 25
    av.prop_separate_params = False
    av.want_cuda = True
    av.model_loc = model_loc
    av.type = ad.type
    model_state_dict = saved['model_state_dict']
    get_result(av,model_loc,model_state_dict)