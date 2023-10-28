import os
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


def save_model_at_epoch(av,model, epoch):
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

def save_initial_model(av,model):
  """
    :param av            : args
    :param model         : nn model whose state_dict is to be saved
    :return              : None
  """
  save_dir = os.path.join(av.DIR_PATH, "initialModels")
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  name = av.DATASET_NAME
  if av.TASK !="":
    name = av.TASK + "_" + name + "_" + str(av.SEED)
  save_prefix = os.path.join(save_dir, name)
  #save_path = '{}_epoch_{}'.format(save_prefix, epoch)

  logger.info("saving initial model to %s",save_prefix)
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



