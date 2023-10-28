import os
from datetime import datetime
import torch
import colorsys
import numpy as np
import torch.nn as nn
import networkx as nx
from common import logger
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

def adjacency_matrix_from_batched_data(idx, batch_data_sizes, from_idx, to_idx, max_set_size):
  batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
  corpus_min, corpus_max = sum(batch_data_sizes_flat[:2*idx+1]), sum(batch_data_sizes_flat[:2*idx+2])  
  query_min, query_max = sum(batch_data_sizes_flat[:2*idx]), sum(batch_data_sizes_flat[:2*idx+1])
  def get_adjacency_matrix(min_idx, max_idx):
    adjacency_matrix = np.zeros((max_set_size, max_set_size))
    for edge_idx, from_node in enumerate(from_idx):
      if from_node >= min_idx and from_node < max_idx:
        adjacency_matrix[from_node - min_idx, to_idx[edge_idx] - min_idx] = 1
    return adjacency_matrix + adjacency_matrix.T
  return get_adjacency_matrix(query_min, query_max), get_adjacency_matrix(corpus_min, corpus_max)

def choose_colors(num_colors):
  np.random.seed(42)
  colors=[]
  for i in np.arange(0., 360., 360. / num_colors):
    hue = i/360.
    lightness = (30 + np.random.rand() * 70)/100.0
    saturation = (30 + np.random.rand() * 70)/100.0
    colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
  return np.array(colors)

def plot_permuted_graphs(query_adj_matrix, corpus_adj_matrix, corpus_permutation, exp_name):
  color_set = choose_colors(len(query_adj_matrix))
  def plot_graph(graph_name, adj_matrix, indexing):
    graph = nx.from_numpy_array(adj_matrix)
    layout = graphviz_layout(graph, prog='neato')  # You can try different layouts here
    nx.draw(graph, pos = layout, with_labels=True, node_color=color_set[indexing])
    loc = f'{exp_name}_{graph_name}.png'
    print(graph_name, "saved to", loc)
    plt.savefig(loc)
    plt.clf()
  plot_graph('query', query_adj_matrix, np.arange(len(query_adj_matrix)))
  corpus_color_indexing = np.zeros(len(corpus_adj_matrix), dtype=np.uint32)
  corpus_color_indexing[corpus_permutation] = np.arange(len(corpus_adj_matrix))
  plot_graph('corpus', corpus_adj_matrix, corpus_color_indexing)

def load_model(av):
  """
  """
  #TODO

def save_model(av,model,optimizerPerm, optimizerFunc, epoch, saveAllEpochs = True):
  """
  """
  #TODO``


def pairwise_ranking_loss(predPos, predNeg, margin):
    
    n_1 = predPos.shape[0]
    n_2 = predNeg.shape[0]
    dim = predPos.shape[1]

    expanded_1 = predPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_2 = predNeg.unsqueeze(0).expand(n_1, n_2, dim)
    ell = margin + expanded_2 - expanded_1
    hinge = nn.ReLU()
    loss = hinge(ell)
    sum_loss =  torch.sum(loss,dim= [0, 1])
    return sum_loss/(n_1*n_2)
    #return sum_loss


def load_model_at_epoch(av,epoch,seed=None):
  """
    :param av           : args
    :return checkpoint  : dict containing model state dicts and av  
  """
  load_dir = os.path.join(av.DIR_PATH, "savedModels")
  if not os.path.isdir(load_dir):
    os.makedirs(load_dir)
    #raise Exception('{} does not exist'.format(load_dir))
  name = av.DATASET_NAME
  seed_suffix = "" if seed is None else f"_{str(seed)}"
  if av.TASK !="":
    name = av.TASK + "_" + name + seed_suffix
  load_prefix = os.path.join(load_dir, name)
  load_path = '{}_epoch_{}'.format(load_prefix, epoch)
  if os.path.exists(load_path):
    logger.info("loading model from %s",load_path)
    checkpoint = torch.load(load_path)
  else: 
    checkpoint= None
  return checkpoint


def save_model_at_epoch(av, model, epoch):
  """
    :param av            : args
    :param model         : nn model whose state_dict is to be saved
    :param epoch         : epoch no. 
    :return              : None
  """
  save_dir = os.path.join(av.DIR_PATH, "savedModels")
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  name = av.DATASET_NAME
  if av.TASK !="":
    name = av.TASK + "_" + name + "_" + str(av.SEED)
  save_prefix = os.path.join(save_dir, name)
  save_path = '{}_epoch_{}'.format(save_prefix, epoch)

  logger.info("saving model to %s",save_path)
  output = open(save_path, mode="wb")
  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'av' : av,
            }, output)
  output.close()

def save_initial_model(av, model):
  """
    :param av            : args
    :param model         : nn model whose state_dict is to be saved
    :return              : None
  """
  save_dir = os.path.join(av.DIR_PATH, av.experiment_group + "/initialModels")
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  name = av.TASK + "_" + av.DATASET_NAME + "_" + str(av.SEED) + "_" + av.time_key
  save_prefix = os.path.join(save_dir, name)

  logger.info("saving initial model to %s", save_prefix)
  output = open(save_prefix, mode="wb")
  torch.save({
            'model_state_dict': model.state_dict(),
            'av' : av,
            }, output)
  output.close()


def cudavar(av, x):
    """Adapt to CUDA or CUDA-less runs.  Annoying av arg may become
    useful for multi-GPU settings."""
    return x.cuda() if av.has_cuda and av.want_cuda else x

def pytorch_sample_gumbel(av,shape, eps=1e-20):
  #Sample from Gumbel(0, 1)
  U = cudavar(av,torch.rand(shape).float())
  return -torch.log(eps - torch.log(U + eps))



def test_indra_our(av, config):
  device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
  train_data = OurMatchingModelSubgraphIsoData(av,mode="train")
  av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
  import copy
  copy_config = copy.deepcopy(config)
  model_our = GMN_embed_with_MLP_and_ColBERT_scores(av,copy_config,1).to(device)
  model_indra = GMN_embed_maxsim_dot_corrected(av,config,1).to(device)

  updated_state_dict = model_our.state_dict().copy()
  updated_state_dict["aggregator.MLP1.0.weight"] = updated_state_dict["node_feature_processor.MLP.0.weight"]
  updated_state_dict["aggregator.MLP2.0.weight"] = model_indra.state_dict()["aggregator.MLP2.0.weight"]
  updated_state_dict["aggregator.MLP1.0.bias"] = updated_state_dict["node_feature_processor.MLP.0.bias"]
  updated_state_dict["aggregator.MLP2.0.bias"] = model_indra.state_dict()["aggregator.MLP2.0.bias"]
  updated_state_dict.pop("node_feature_processor.MLP.0.weight")
  updated_state_dict.pop("node_feature_processor.MLP.0.bias")
  model_indra.load_state_dict(updated_state_dict)

  torch.save(model_our.state_dict(), "initialModelWeights/matching_iso_var_gmn_with_mlp_and_colbert_objective")
  torch.save(model_indra.state_dict(), "initialModelWeights/matching_iso_var_gmn_with_maxsim_dot_corrected")

  exit()

  train_data.data_type = "gmn"
  for batch_idx in range(1000):
    batch_data, batch_data_sizes, _, batch_adj = train_data.fetch_batched_data_by_id(batch_idx)
    scores_our = model_our(batch_data,batch_data_sizes,batch_adj)
    scores_indra = model_indra(batch_data,batch_data_sizes,batch_adj)
    score_delta = torch.sum((scores_our - scores_indra)**2)
    print(scores_our, scores_indra)
    if score_delta > 1e-10:
      print("Mismatch found", score_delta)
      exit(0)

def compare_hungarian_with_normal(av, config):
  device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
  train_data = OurMatchingModelSubgraphIsoData(av,mode="train")
  av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
  import copy
  copy_config = copy.deepcopy(config)
  model_1 = Node_align_Node_loss(av,copy_config,1).to(device)
  model_2 = Hungarian_Node_align_Node_loss(av,config,1).to(device)

  model_1_state_dict_copy = model_1.state_dict().copy()
  model_2_state_dict_copy = model_2.state_dict().copy()
  for key in model_2_state_dict_copy.keys():
    if "asymm_" in key:
      model_2_state_dict_copy[key] = model_1_state_dict_copy[key.replace("asymm_", "")]
  model_2.load_state_dict(model_2_state_dict_copy)

  n_batches = train_data.create_stratified_batches()
  train_data.data_type = "gmn"
  for batch_idx in range(1000):
    model_1.train()
    model_2.train()
    batch_data, batch_data_sizes, _, batch_adj = train_data.fetch_batched_data_by_id(batch_idx)
    scores_1 = model_1(batch_data,batch_data_sizes,batch_adj)
    scores_2 = model_2(batch_data,batch_data_sizes,batch_adj)
    score_delta = torch.sum((scores_1 - scores_2)**2)
    if score_delta > 1e-10:
      # Mismatch because of randomness in sinkhorn_iters
      print("Mismatch found", batch_idx, score_delta)
      exit(0)

def plot_hungarian_graph_pairs(av, config):
  device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
  train_data = OurMatchingModelSubgraphIsoData(av,mode="train")
  av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                   max([g.number_of_edges() for g in train_data.corpus_graphs]))
  import copy
  copy_config = copy.deepcopy(config)
  hungarian_model = Hungarian_Node_align_Node_loss(av,copy_config,1).to(device)
  hungarian_model.load_state_dict(torch.load("/mnt/home/ashwinr/btp24/grph/ISONET/bestValidationModels/hungarian_node_align_node_loss_truncated_transport_plan_ptc_fm_0")['model_state_dict'])

  n_batches = train_data.create_stratified_batches()
  train_data.data_type = "gmn"
  for batch_idx in range(1000):
    batch_data, batch_data_sizes, _, batch_adj = train_data.fetch_batched_data_by_id(batch_idx)
    hungarian_model.visualize([0, 1, 2, 3],batch_data,batch_data_sizes,batch_adj)
    input()
