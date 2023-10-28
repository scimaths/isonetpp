import argparse
import sys
import scipy
import torch
import matplotlib.pyplot as plt
import colorsys

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import torch.nn as nn
import pickle
from common import logger, set_log
import networkx as nx
import random 
import math
import numpy as np
from subgraph.utils import cudavar,save_initial_model, save_model_at_epoch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import os
from sklearn.metrics import average_precision_score
import time
from subgraph.earlystopping import EarlyStoppingModule
#import copy
from torch.nn.utils.rnn import  pad_sequence 
from subgraph.graphs import TUDatasetGraph
import networkx as nx

import torch.nn.functional as F
import collections
import GMN.utils as gmnutils
import GMN.graphembeddingnetwork as gmngen
import GMN.graphmatchingnetwork as gmngmn
from GMN.configure import *
from GMN.loss import euclidean_distance
import GraphOTSim.python.layers as gotsim_layers
from lap import lapjv
from lap import lapmod
import subgraph.neuromatch as nm

from datetime import datetime

from subgraph.iso_matching_models import OurMatchingModelSubgraphIsoData, pytorch_sample_gumbel, pytorch_sinkhorn_iters, seed_everything

total_count = 0
mismatch_count = 0

class Node_align_Node_loss(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(Node_align_Node_loss, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
        
    def build_masking_utility(self):
        self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))).cuda() for x in range(0,self.max_set_size+1)]
        
   

    def build_layers(self):

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        
        #NOTE:FILTERS_3 is 10 for now - hardcoded into config
        self.fc_transform1 = torch.nn.Linear(self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def visualize(self, index, transport_plan, batch_data_sizes, to_idx, from_idx):
        # (q, c)
        transport_plan = transport_plan[index].detach().to('cpu').numpy()
        from_idx_dupl = from_idx.detach().to('cpu').numpy()
        to_idx_dupl = to_idx.detach().to('cpu').numpy()
        batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
        
        # logs
        # print("Transport plan", np.array_str(transport_plan, precision=3, suppress_small=False))
        # print("Row sum of transport plan", transport_plan.sum(axis=1))
        # print("Column sum of transport plan", transport_plan.sum(axis=0))
        print("Shape of transport plan", transport_plan.shape)
        row_argmax = np.argmax(transport_plan, 1)
        print("Row argmax of transport plan", row_argmax)
        print("Graph sizes", batch_data_sizes[index])
        q_size = batch_data_sizes[index][0]
        c_size = batch_data_sizes[index][1]

        hung_q, hung_c = scipy.optimize.linear_sum_assignment(-transport_plan)
        hung_alignment = np.arange(15)
        print("Hungarian indices")
        print(hung_q)
        print(hung_c)
        hung_alignment[hung_c] = hung_q
        if c_size < 15:
            hung_mismatch = np.min(hung_alignment[c_size:]) < q_size
        else:
            hung_mismatch = 0
        print("Flipped Hungarian Alignment", hung_alignment, hung_mismatch)

        # design the permutation matrix
        permutation = np.argmax(transport_plan, axis=1)
        permutation_matrix = np.zeros((self.max_set_size, self.max_set_size))
        for i in range(self.max_set_size):
            permutation_matrix[i, permutation[i]] = 1

        def choose_colors(num_colors):
            np.random.seed(42)
            colors=[]
            for i in np.arange(0., 360., 360. / num_colors):
                hue = i/360.
                lightness = (30 + np.random.rand() * 70)/100.0
                saturation = (30 + np.random.rand() * 70)/100.0
                colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
            return np.array(colors)

        set_col = choose_colors(self.max_set_size)
        def get(index_qc):
            if index_qc:
                min_num = sum(batch_data_sizes_flat[ : 2 * index + 1])
                max_num = sum(batch_data_sizes_flat[ : 2 * index + 2])
                indexing = hung_q
            else:
                min_num = sum(batch_data_sizes_flat[ : 2 * index ])
                max_num = sum(batch_data_sizes_flat[ : 2 * index + 1])
                indexing = hung_c
            print("Visualizing min_num", min_num, "max_num", max_num)
        
            # adjacency_matrix = [[0 for i in range(batch_data_sizes[index][index_qc])] for j in range(batch_data_sizes[index][index_qc])]
            adjacency_matrix = [[0 for i in range(self.max_set_size)] for j in range(self.max_set_size)]
        
            for i in range(len(from_idx_dupl)):
                if from_idx_dupl[i] >= min_num and from_idx_dupl[i] < max_num:
                    adjacency_matrix[from_idx_dupl[i] - min_num][to_idx_dupl[i] - min_num] = 1
                    adjacency_matrix[to_idx_dupl[i] - min_num][from_idx_dupl[i] - min_num] = 1
                    print("Edge -", from_idx_dupl[i], to_idx_dupl[i])
        
            adjacency_matrix = np.array(adjacency_matrix)
        
            # if index_qc:
            #     adjacency_matrix = permutation_matrix @ adjacency_matrix @ permutation_matrix.transpose()
            #     num_nodes_q = batch_data_sizes[index][0]
            #     num_nodes_c = batch_data_sizes[index][1]
            #     transport_plan_here = transport_plan[index].detach().to('cpu').numpy()
            #     argmax_indices = np.argmax(transport_plan_here, axis=1)
            #     filter_indices = argmax_indices[:num_nodes_q]
            #     print(filter_indices)
            #     adjacency_matrix = adjacency_matrix[filter_indices, :][:, filter_indices]
            from networkx.drawing.nx_agraph import graphviz_layout

            graph = nx.from_numpy_array(adjacency_matrix)
            layout = graphviz_layout(graph, prog='neato')  # You can try different layouts here
            nx.draw(graph, pos = layout, with_labels=True, node_color=set_col[indexing])
            loc = 'index_node_align_unaligned_' + str(index) + '_' + str(index_qc) + '.png'
            print(loc)
            plt.savefig(loc)
            plt.clf()
        if not hung_mismatch:
            get(0)
            get(1)
            input()
        return hung_mismatch
        
    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
        """
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))
        #A
        a, b = zip(*batch_adj)
        q_adj = torch.stack(a)
        c_adj = torch.stack(b)

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)

        #[(8, 12), (10, 13), (10, 14)] -> [8, 12, 10, 13, 10, 14]
        batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]
        assert(list(zip([x.shape[0] for x in node_feature_enc_query], \
                        [x.shape[0] for x in node_feature_enc_corpus])) \
               == batch_data_sizes)        
        
        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in node_feature_enc_corpus])

        transformed_qnode_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qnode_emb)))
        transformed_cnode_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cnode_emb)))
        qgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_sizes]))
        cgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_sizes]))
        masked_qnode_emb = torch.mul(qgraph_mask,transformed_qnode_emb)
        masked_cnode_emb = torch.mul(cgraph_mask,transformed_cnode_emb)
 
        sinkhorn_input = torch.matmul(masked_qnode_emb,masked_cnode_emb.permute(0,2,1))
        # transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input, n_iters=2000)
        transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input, n_iters=20)

        if self.diagnostic_mode:
            return transport_plan
        
        scores = -torch.sum(torch.maximum(stacked_qnode_emb - transport_plan@stacked_cnode_emb,\
              cudavar(self.av,torch.tensor([0]))),\
           dim=(1,2))

        mismatches = 0
        total = 0

        # visualize
        scores = scores.detach().to('cpu').numpy()
        print(scores.shape)
        indices_descending = np.argsort(scores)[::-1]
        for index in indices_descending:
            total += 1
            print("Scores", scores[index])
            mismatches += self.visualize(index, transport_plan, batch_data_sizes, to_idx, from_idx)

        return mismatches, total

class Fringed_node_align_Node_loss(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(Fringed_node_align_Node_loss, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
        
    def build_masking_utility(self):
        self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_set_size+1)]
        
    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        
        #NOTE:FILTERS_3 is 10 for now - hardcoded into config
        print(self.av.transform_dim, self.av.filters_3)
        self.fc_transform1 = torch.nn.Linear(self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
        """
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))

        # initially corpus is not masked
        mask_from_idx = None

        # this iterations decide how much the masking should be
        for j in range(5):
          node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
          node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)

          for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
              node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc,mask_from_idx=mask_from_idx)

          #[(8, 12), (10, 13), (10, 14)] -> [8, 12, 10, 13, 10, 14]
          batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
          node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
          node_feature_enc_query = node_feature_enc_split[0::2]
          node_feature_enc_corpus = node_feature_enc_split[1::2]   
          
          stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                          for x in node_feature_enc_query])
          stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                          for x in node_feature_enc_corpus])

          transformed_qnode_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qnode_emb)))
          transformed_cnode_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cnode_emb)))
          qgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_sizes]))
          cgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_sizes]))
          masked_qnode_emb = torch.mul(qgraph_mask,transformed_qnode_emb)
          masked_cnode_emb = torch.mul(cgraph_mask,transformed_cnode_emb)
  
          sinkhorn_input = torch.matmul(masked_qnode_emb,masked_cnode_emb.permute(0,2,1))
          transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input)
          
          # N - number of tuples (query, corpus)
          # M - maximum number of nodes in a graph
          # D - embedding dimension

          # Slice QueryMask(N, M, D) to (N, M, M)
          # QueryMask has ones where the query nodes are present and 0s in
          
          temp_mask_from_idx = torch.sum(transport_plan * qgraph_mask[:, :, :self.max_set_size], dim=2)
          temp_mask_from_idx = torch.stack((cudavar(self.av, torch.ones(temp_mask_from_idx.shape)), temp_mask_from_idx), dim = 1).view( temp_mask_from_idx.shape[0] * 2, temp_mask_from_idx.shape[1]).flatten()
          mask_from_idx = torch.cat(torch.split(temp_mask_from_idx, torch.stack((torch.tensor(batch_data_sizes_flat), self.max_set_size - torch.tensor(batch_data_sizes_flat)), dim = 1).view(2 * len(batch_data_sizes_flat)).tolist(), dim=0)[0::2])
          
          print(from_idx, to_idx)
          print(np.array_str(transport_plan[0].detach().to('cpu').numpy(), precision=3, suppress_small=False))
          print(transport_plan[0].sum(dim=1))
          print(transport_plan[0].sum(dim=0))
          print(transport_plan.shape)
          print(batch_data_sizes)
        
        if self.diagnostic_mode:
            return transport_plan
        
        scores = -torch.sum(torch.maximum(stacked_qnode_emb - transport_plan@stacked_cnode_emb,\
              cudavar(self.av,torch.tensor([0]))),\
          dim=(1,2))
        
        return scores

class ISONET(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(ISONET, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
        
    def build_masking_utility(self):
        self.max_set_size = self.av.MAX_EDGES
        #this mask pattern sets bottom last few rows to 0 based on padding needs
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_set_size+1)]
        # Mask pattern sets top left (k)*(k) square to 1 inside arrays of size n*n. Rest elements are 0
        self.set_size_to_mask_map = [torch.cat((torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([x,self.max_set_size-x])).repeat(x,1),
                             torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([0,self.max_set_size])).repeat(self.max_set_size-x,1)))
                             for x in range(0,self.max_set_size+1)]

        
    def fetch_edge_counts(self,to_idx,from_idx,graph_idx,num_graphs):
        #HACK - since I'm not storing edge sizes of each graph (only storing node sizes)
        #and no. of nodes is not equal to no. of edges
        #so a hack to obtain no of edges in each graph from available info
        from GMN.segment import unsorted_segment_sum
        tt = unsorted_segment_sum(cudavar(self.av,torch.ones(len(to_idx))), to_idx, len(graph_idx))
        tt1 = unsorted_segment_sum(cudavar(self.av,torch.ones(len(from_idx))), from_idx, len(graph_idx))
        edge_counts = unsorted_segment_sum(tt, graph_idx, num_graphs)
        edge_counts1 = unsorted_segment_sum(tt1, graph_idx, num_graphs)
        assert(edge_counts == edge_counts1).all()
        assert(sum(edge_counts)== len(to_idx))
        return list(map(int,edge_counts.tolist()))

    def build_layers(self):

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        
        #NOTE:FILTERS_3 is 10 for now - hardcoded into config
        self.fc_transform1 = torch.nn.Linear(2*self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        
        #self.edge_score_fc = torch.nn.Linear(self.prop_layer._message_net[-1].out_features, 1)
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    
    
    def visualize(self, index, transport_plan, batch_data_sizes, to_idx, from_idx):
        # (q, c)
        transport_plan = transport_plan[index].detach().to('cpu').numpy()
        from_idx_dupl = from_idx.detach().to('cpu').numpy()
        to_idx_dupl = to_idx.detach().to('cpu').numpy()
        batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
        
        # logs
        print(np.array_str(transport_plan, precision=3, suppress_small=False))
        print(transport_plan.sum(axis=1))
        print(transport_plan.sum(axis=0))
        print(transport_plan.shape)
        print(np.argmax(transport_plan, 1))
        print(batch_data_sizes[index])

        # design the permutation matrix
        permutation = np.argmax(transport_plan, axis=1)
        permutation_matrix = np.zeros((self.max_set_size, self.max_set_size))
        for i in range(self.max_set_size):
            permutation_matrix[i, permutation[i]] = 1
        def choose_colors(num_colors):
                np.random.seed(42)
                colors=[]
                for i in np.arange(0., 360., 360. / num_colors):
                    hue = i/360.
                    lightness = (30 + np.random.rand() * 70)/100.0
                    saturation = (30 + np.random.rand() * 70)/100.0
                    colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
                return colors
        set_col = choose_colors(self.max_set_size)
        def get(index_qc):
            if index_qc:
                min_num = sum(batch_data_sizes_flat[ : 2 * index + 1])
                max_num = sum(batch_data_sizes_flat[ : 2 * index + 2])
            else:
                min_num = sum(batch_data_sizes_flat[ : 2 * index ])
                max_num = sum(batch_data_sizes_flat[ : 2 * index + 1])
        
            # adjacency_matrix = [[0 for i in range(batch_data_sizes[index][index_qc])] for j in range(batch_data_sizes[index][index_qc])]
            adjacency_matrix = [[0 for i in range(self.max_set_size)] for j in range(self.max_set_size)]
        
            for i in range(len(from_idx_dupl)):
                if from_idx_dupl[i] >= min_num and from_idx_dupl[i] < max_num:
                    adjacency_matrix[from_idx_dupl[i] - min_num][to_idx_dupl[i] - min_num] = 1
                    adjacency_matrix[to_idx_dupl[i] - min_num][from_idx_dupl[i] - min_num] = 1
        
            adjacency_matrix = np.array(adjacency_matrix)
        
            if index_qc:
                adjacency_matrix = permutation_matrix @ adjacency_matrix @ permutation_matrix.transpose()
            #     num_nodes_q = batch_data_sizes[index][0]
            #     num_nodes_c = batch_data_sizes[index][1]
            #     transport_plan_here = transport_plan[index].detach().to('cpu').numpy()
            #     argmax_indices = np.argmax(transport_plan_here, axis=1)
            #     filter_indices = argmax_indices[:num_nodes_q]
            #     print(filter_indices)
            #     adjacency_matrix = adjacency_matrix[filter_indices, :][:, filter_indices]
            from networkx.drawing.nx_agraph import graphviz_layout

            graph = nx.from_numpy_array(adjacency_matrix)
            layout = graphviz_layout(graph, prog='neato')  # You can try different layouts here
            nx.draw(graph, pos = layout, with_labels=True, node_color=set_col)
            plt.savefig('index_isonet_' + str(index) + '_' + str(index_qc) + '.png')
            plt.clf()

        get(0)
        get(1)


    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
        """
        #a,b = zip(*batch_data_sizes)
        #qgraph_sizes = cudavar(self.av,torch.tensor(a))
        #cgraph_sizes = cudavar(self.av,torch.tensor(b))
        #A
        #a, b = zip(*batch_adj)
        #q_adj = torch.stack(a)
        #c_adj = torch.stack(b)
        

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            #node_feature_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            
        source_node_enc = node_features_enc[from_idx]
        dest_node_enc  = node_features_enc[to_idx]
        forward_edge_input = torch.cat((source_node_enc,dest_node_enc,edge_features_enc),dim=-1)
        backward_edge_input = torch.cat((dest_node_enc,source_node_enc,edge_features_enc),dim=-1)
        forward_edge_msg = self.prop_layer._message_net(forward_edge_input)
        backward_edge_msg = self.prop_layer._reverse_message_net(backward_edge_input)
        edge_features_enc = forward_edge_msg + backward_edge_msg
        
        edge_counts  = self.fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
        qgraph_edge_sizes = cudavar(self.av,torch.tensor(edge_counts[0::2]))
        cgraph_edge_sizes = cudavar(self.av,torch.tensor(edge_counts[1::2]))

        edge_feature_enc_split = torch.split(edge_features_enc, edge_counts, dim=0)
        edge_feature_enc_query = edge_feature_enc_split[0::2]
        edge_feature_enc_corpus = edge_feature_enc_split[1::2]  
        
        
        stacked_qedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in edge_feature_enc_query])
        stacked_cedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in edge_feature_enc_corpus])


        transformed_qedge_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qedge_emb)))
        transformed_cedge_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cedge_emb)))
        qgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_edge_sizes]))
        cgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_edge_sizes]))
        masked_qedge_emb = torch.mul(qgraph_mask,transformed_qedge_emb)
        masked_cedge_emb = torch.mul(cgraph_mask,transformed_cedge_emb)
 
        sinkhorn_input = torch.matmul(masked_qedge_emb,masked_cedge_emb.permute(0,2,1))
        transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input)
        # which corpus nodes to be unmatched by query -- need this only with soft permutation


 
        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum(torch.maximum(stacked_qedge_emb - transport_plan@stacked_cedge_emb,\
              cudavar(self.av,torch.tensor([0]))),\
           dim=(1,2))
        
        # visualize
        scores = scores.detach().to('cpu').numpy()
        print(scores.shape)
        indices_descending = np.argsort(scores)[::-1]
        for index in indices_descending:
            # index = 1
            print(scores[index])
            self.visualize(index, transport_plan, batch_data_sizes, to_idx, from_idx)
            input()
        return scores

def test_node_align_node_loss(av, config):
    device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
    train_data = OurMatchingModelSubgraphIsoData(av,mode="train")
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                    max([g.number_of_edges() for g in train_data.corpus_graphs]))

    es = EarlyStoppingModule(av)

    # model = Fringed_node_align_Node_loss(av, config, 1)
    # model = ISONET(av, config, 1).to(device)
    model = Node_align_Node_loss(av, config, 1).to(device)
    model.load_state_dict(es.load_best_model()['model_state_dict'])
    model.to(device)

    train_data.data_type = "gmn"

    n_batches = train_data.create_stratified_batches()

    total = 0
    mismatches = 0
    print(train_data.num_batches)
    input()
    for batch_idx in range(train_data.num_batches):
        batch_data, batch_data_sizes, _, batch_adj = train_data.fetch_batched_data_by_id(batch_idx)
        m, t = model(batch_data,batch_data_sizes,batch_adj)
        mismatches += m
        total += t
        print("Batch-IDX", batch_idx, mismatches, total)

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--logpath",                        type=str,   default="logDir/logfile",help="/path/to/log")
  ap.add_argument("--want_cuda",                      type=bool,  default=True)
  ap.add_argument("--RUN_TILL_ES",                    type=bool,  default=True)
  ap.add_argument("--has_cuda",                       type=bool,  default=torch.cuda.is_available())
  ap.add_argument("--is_sig",                         type=bool,  default=False)
  ap.add_argument("--ES",                             type=int,   default=50)
  ap.add_argument("--MIN_QUERY_SUBGRAPH_SIZE",        type=int,   default=5)
  ap.add_argument("--MAX_QUERY_SUBGRAPH_SIZE",        type=int,   default=10)
  ap.add_argument("--MIN_CORPUS_SUBGRAPH_SIZE",       type=int,   default=11)
  ap.add_argument("--MAX_CORPUS_SUBGRAPH_SIZE",       type=int,   default=15)
  ap.add_argument("--MAX_GRAPH_SIZE",                 type=int,   default=0)
  ap.add_argument("--n_layers",                       type=int,   default=3)
  ap.add_argument("--conv_type",                      type=str,   default='SAGE')
  ap.add_argument("--method_type",                    type=str,   default='order')
  ap.add_argument("--skip",                           type=str,   default='learnable')
  ap.add_argument("--neuromatch_hidden_dim",          type=int,   default=10)
  ap.add_argument("--post_mp_dim",                    type=int,   default=64)
  ap.add_argument("--filters_1",                      type=int,   default=128)
  ap.add_argument("--filters_2",                      type=int,   default=64)
  ap.add_argument("--filters_3",                      type=int,   default=10)
  ap.add_argument("--dropout",                        type=float, default=0)
  ap.add_argument("--tensor_neurons",                 type=int,   default=16)
  ap.add_argument("--transform_dim" ,                 type=int,   default=10)
  ap.add_argument("--bottle_neck_neurons",            type=int,   default=16)
  ap.add_argument("--bins",                           type=int,   default=16)
  ap.add_argument("--histogram",                      type=bool,  default=False)
  ap.add_argument("--GMN_NPROPLAYERS",                type=int,   default=5)
  ap.add_argument("--MARGIN",                         type=float, default=0.1)
  ap.add_argument("--NOISE_FACTOR",                   type=float, default=1.0)
  ap.add_argument("--NUM_RUNS",                       type=int,   default=2)
  ap.add_argument("--BATCH_SIZE",                     type=int,   default=128)
  ap.add_argument("--LEARNING_RATE",                  type=float, default=0.001)
  ap.add_argument("--WEIGHT_DECAY",                   type=float, default=5*10**-4)
  ap.add_argument("--FEAT_TYPE",                      type=str,   default="Onehot1",help="One/Onehot/Onehot1/Adjrow/Adjrow1/AdjOnehot")
  ap.add_argument("--CONV",                           type=str,   default="GCN",help="GCN/GAT/GIN/SAGE")
  ap.add_argument("--DIR_PATH",                       type=str,   default=".",help="path/to/datasets")
  ap.add_argument("--DATASET_NAME",                   type=str,   default="mutag", help="TODO")
  ap.add_argument("--TASK",                           type=str,   default="OurMatchingSimilarity",help="TODO")
  ap.add_argument("--SEED",                           type=int,   default=0)

  av = ap.parse_args()

  if av.FEAT_TYPE == "Adjrow" or  av.FEAT_TYPE == "Adjrow1" or av.FEAT_TYPE == "AdjOnehot": 
      av.TASK = av.TASK + "_" + av.FEAT_TYPE
  if av.CONV != "GCN": 
      av.TASK = av.TASK + "_" + av.CONV
#   av.logpath = av.logpath+"_"+av.TASK+"_"+av.DATASET_NAME+"_"+str(av.SEED)+"_"+str(datetime.now()).replace(" ", "_")
  av.logpath = av.logpath+"_"+av.TASK+"_"+av.DATASET_NAME+"_"+str(datetime.now()).replace(" ", "_")
#   set_log(av)

  # Print configure
  config = get_default_config()
  config['seed'] = av.SEED
  config['encoder'] ['node_hidden_sizes'] = [10]
  config['encoder'] ['node_feature_dim'] = 1
  config['encoder'] ['edge_feature_dim'] = 1
  config['aggregator'] ['node_hidden_sizes'] = [10]
  config['aggregator'] ['graph_transform_sizes'] = [10]
  config['aggregator'] ['input_size'] = [10]
  config['graph_matching_net'] ['node_state_dim'] = 10
  #config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_matching_net'] ['edge_hidden_sizes'] = [20]
  config['graph_matching_net'] ['node_hidden_sizes'] = [10]
  config['graph_matching_net'] ['n_prop_layers'] = 5
  config['graph_embedding_net'] ['node_state_dim'] = 10
  #config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_embedding_net'] ['edge_hidden_sizes'] = [20]
  config['graph_embedding_net'] ['node_hidden_sizes'] = [10]
  config['graph_embedding_net'] ['n_prop_layers'] = 5
  
  config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS

  config['training']['batch_size']  = av.BATCH_SIZE
  config['training']['margin']  = av.MARGIN
  config['evaluation']['batch_size']  = av.BATCH_SIZE
  config['model_type']  = "embedding"
  config['graphsim'] = {} 
  config['graphsim']['conv_kernel_size'] = [10,4,2]
  config['graphsim']['linear_size'] = [24, 16]
  config['graphsim']['gcn_size'] = [10,10,10]
  config['graphsim']['conv_pool_size'] = [3,3,2]
  config['graphsim']['conv_out_channels'] = [2,4,8]
  config['graphsim']['dropout'] = av.dropout 

  # Set random seeds
  seed = config['seed']
  seed_everything(seed)

  av.dataset = av.DATASET_NAME
  test_node_align_node_loss(av, config)


