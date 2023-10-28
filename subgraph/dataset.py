import math
import torch
import pickle
import random
import numpy as np
import collections
import networkx as nx
from common import logger
import torch.nn.functional as F
from subgraph.utils import cudavar
from torch_geometric.data import Data

class OurMatchingModelSubgraphIsoData(object):
  """
  """
  def __init__(self,av,mode="train"):
    self.av = av
    self.mode = mode
    self.load_graphs()
    self.preprocess_subgraphs_to_pyG_data()
    self.fetch_subgraph_adjacency_info()
    self.data_type = "pyg" #"pyg"/"gmn"

  def load_graphs(self):
    """
      self.query_graphs : list of nx graphs 
      self.corpus_graphs : list of nx graphs
      self.rels : dict denoting 'pos'/'neg' 
      self.list_pos
      self.list_neg
    """

    #LOAD query graphs in data split
    if self.mode == "Extra_test_300":
        fp = self.av.DIR_PATH + "/Datasets/splits/Extra_test_300/test_" +self.av.DATASET_NAME +"_80k_query_subgraphs.pkl"
    else:
        fp = self.av.DIR_PATH + "/Datasets/splits/" + self.mode + "/" + self.mode + "_" +\
                self.av.DATASET_NAME +"80k_query_subgraphs.pkl"

    self.query_graphs = pickle.load(open(fp,"rb"))
    logger.info("loading %s query graphs from %s", self.mode, fp)

    #LOAD iso relaitonships of query graphs wrt corpus graphs
    if self.mode == "Extra_test_300":
        fp = self.av.DIR_PATH + "/Datasets/splits/Extra_test_300/test_" +self.av.DATASET_NAME + "_80k_rel_nx_is_subgraph_iso.pkl"
    else:
        fp = self.av.DIR_PATH + "/Datasets/splits/" + self.mode + "/" + self.mode + "_" +\
                self.av.DATASET_NAME + "80k_rel_nx_is_subgraph_iso.pkl"
    self.rels = pickle.load(open(fp,"rb"))
    logger.info("loading %s relationships from %s", self.mode, fp)

    #LOAD all corpus graphs
    fp = self.av.DIR_PATH + "/Datasets/splits/" + self.av.DATASET_NAME +"80k_corpus_subgraphs.pkl"
    self.corpus_graphs = pickle.load(open(fp,"rb"))
    logger.info("loading corpus graphs from %s", fp)
    assert(list(range(len(self.query_graphs))) == list(self.rels.keys()))

    
    self.list_pos = []
    self.list_neg = []   

    for q in range(len(self.query_graphs)) :
      for c in self.rels[q]['pos']:
        self.list_pos.append(((q,c),1.0))
      for c in self.rels[q]['neg']:
        self.list_neg.append(((q,c),0.0))  
        
  def create_pyG_data_object(self,g):
    if self.av.FEAT_TYPE == "One":
      #This sets node features to one aka [1]
      x1 = cudavar(self.av,torch.FloatTensor(torch.ones(g.number_of_nodes(),1)))
    else:
      raise NotImplementedError()  
      
    l = list(g.edges)
    edges_1 = [[x,y] for (x,y) in l ]+ [[y,x] for (x,y) in l]
    edge_index = cudavar(self.av,torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long))
    return Data(x=x1,edge_index=edge_index),g.number_of_nodes()
  
  def preprocess_subgraphs_to_pyG_data(self):
    """
      self.query_graph_data_list
      self.query_graph_size_list
      self.corpus_graph_data_list
      self.corpus_graph_size_list
    """
    self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
    if self.av.FEAT_TYPE == "One":
        self.num_features = 1
    else:
      self.num_features = self.max_set_size
    self.query_graph_data_list = []
    self.query_graph_size_list = []
    n_graphs = len(self.query_graphs)
    for i in range(n_graphs): 
      data,size = self.create_pyG_data_object(self.query_graphs[i])
      self.query_graph_data_list.append(data)
      self.query_graph_size_list.append(size)

    self.corpus_graph_data_list = []
    self.corpus_graph_size_list = []
    n_graphs = len(self.corpus_graphs)
    for i in range(n_graphs): 
      data,size = self.create_pyG_data_object(self.corpus_graphs[i])
      self.corpus_graph_data_list.append(data)
      self.corpus_graph_size_list.append(size)     

  def fetch_subgraph_adjacency_info(self):
    """
      used for input to hinge scoring
      self.query_graph_adj_list
      self.corpus_graph_adj_list
    """
    self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
    self.query_graph_adj_list = []
    n_graphs = len(self.query_graphs)
    for i in range(n_graphs):
      g = self.query_graphs[i]
      x1 = cudavar(self.av,torch.FloatTensor(nx.adjacency_matrix(g).todense()))
      x2 = F.pad(x1,pad=(0,self.max_set_size-x1.shape[1],0,self.max_set_size-x1.shape[0]))
      self.query_graph_adj_list.append(x2)

    self.corpus_graph_adj_list = []
    n_graphs = len(self.corpus_graphs)
    for i in range(n_graphs): 
      g = self.corpus_graphs[i]
      x1 = cudavar(self.av,torch.FloatTensor(nx.adjacency_matrix(g).todense()))
      x2 = F.pad(x1,pad=(0,self.max_set_size-x1.shape[1],0,self.max_set_size-x1.shape[0]))
      self.corpus_graph_adj_list.append(x2)      

  def _pack_batch(self, graphs):
        """Pack a batch of graphs into a single `GraphData` instance.
    Args:
      graphs: a list of generated networkx graphs.
    Returns:
      graph_data: a `GraphData` instance, with node and edge indices properly
        shifted.
    """
        Graphs = []
        for graph in graphs:
            for inergraph in graph:
                Graphs.append(inergraph)
        graphs = Graphs
        from_idx = []
        to_idx = []
        graph_idx = []

        n_total_nodes = 0
        n_total_edges = 0
        for i, g in enumerate(graphs):
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
            edges = np.array(g.edges(), dtype=np.int32)
            # shift the node indices for the edges
            from_idx.append(edges[:, 0] + n_total_nodes)
            to_idx.append(edges[:, 1] + n_total_nodes)
            graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

            n_total_nodes += n_nodes
            n_total_edges += n_edges

        GraphData = collections.namedtuple('GraphData', [
            'from_idx',
            'to_idx',
            'node_features',
            'edge_features',
            'graph_idx',
            'n_graphs'])

        return GraphData(
            from_idx=np.concatenate(from_idx, axis=0),
            to_idx=np.concatenate(to_idx, axis=0),
            # this task only cares about the structures, the graphs have no features
            node_features=np.ones((n_total_nodes, 1), dtype=np.float32),
            edge_features=np.ones((n_total_edges, 1), dtype=np.float32),
            graph_idx=np.concatenate(graph_idx, axis=0),
            n_graphs=len(graphs),
        )    
    
  def create_stratified_batches(self):
    """
      Creates shuffled batches while maintaining class ratio
    """
    lpos = self.list_pos
    lneg = self.list_neg
    random.shuffle(lpos)
    random.shuffle(lneg)
    p2n_ratio = len(lpos)/len(lneg)
    npos = math.ceil((p2n_ratio/(1+p2n_ratio))*self.av.BATCH_SIZE)
    nneg = self.av.BATCH_SIZE-npos
    batches_pos, batches_neg = [],[]
    for i in range(0, len(lpos), npos):
      batches_pos.append(lpos[i:i+npos])
    for i in range(0, len(lneg), nneg):
      batches_neg.append(lneg[i:i+nneg])
     
    self.num_batches = min(len(batches_pos),len(batches_neg))  
    self.batches = [a+b for (a,b) in zip(batches_pos[:self.num_batches],batches_neg[:self.num_batches])]
    return self.num_batches

  def create_batches(self,list_all):
    """
      create batches as is and return number of batches created
    """

    self.batches = []
    for i in range(0, len(list_all), self.av.BATCH_SIZE):
      self.batches.append(list_all[i:i+self.av.BATCH_SIZE])
   
    self.num_batches = len(self.batches)  

    return self.num_batches
        
  def fetch_batched_data_by_id(self,i):
    """
      all_data  : graph node, edge info
      all_sizes : this is required to create padding tensors for
                  batching variable size graphs
      target    : labels/scores             
    """
    assert(i < self.num_batches)  
    batch = self.batches[i]
    
    a,b = zip(*batch)
    g_pair = list(a)
    score = list(b)
    
    a,b = zip(*g_pair)
    if self.data_type =="gmn":
      g1 = [self.query_graphs[i] for i in a]  
    else:
      g1 = [self.query_graph_data_list[i] for i in a]
    g1_size = [self.query_graph_size_list[i] for i in a]
    g1_adj  = [self.query_graph_adj_list[i]  for i in a]
    if self.data_type =="gmn":
      g2 = [self.corpus_graphs[i] for i in b]      
    else:    
      g2 = [self.corpus_graph_data_list[i] for i in b]
    g2_size = [self.corpus_graph_size_list[i] for i in b]
    g2_adj  = [self.corpus_graph_adj_list[i]  for i in b]
    
    if self.data_type =="gmn":
      all_data = self._pack_batch(zip(g1,g2))
    else:
      all_data = list(zip(g1,g2))
    all_sizes = list(zip(g1_size,g2_size))
    all_adj = list(zip(g1_adj,g2_adj))
    target = cudavar(self.av,torch.tensor(score))
    return all_data, all_sizes, target, all_adj
