import torch
import torch.nn.functional as F
from subgraph.utils import cudavar
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen
from subgraph.models.node_align_node_loss import Node_align_Node_loss

class Fringed_node_align_Node_loss_aliter(Node_align_Node_loss):
    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.FringedGraphPropLayer(**prop_config)
        
        #NOTE:FILTERS_3 is 10 for now - hardcoded into config
        self.fc_transform1 = torch.nn.Linear(self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        self.masking_iters = self.config['fringe_isonet']['masking_for_msg_passing_count']

    def _pad_and_apply_sinkhorn(self, node_features_enc, batch_data_sizes_flat):
        #[(8, 12), (10, 13), (10, 14)] -> [8, 12, 10, 13, 10, 14]
        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]

        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                        for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                        for x in node_feature_enc_corpus])

        sinkhorn_input = torch.matmul(stacked_qnode_emb,stacked_cnode_emb.permute(0,2,1))
        transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input)
        return stacked_qnode_emb, stacked_cnode_emb, transport_plan

    def forward(self, batch_data,batch_data_sizes, batch_adj):
        """
        """
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))

        fringe_weights = torch.ones(len(batch_data_sizes), self.max_set_size, device=qgraph_sizes.device)

        padding_mask = lambda graph_sizes: (torch.arange(self.max_set_size, device=graph_sizes.device)[None, :] < \
                                            graph_sizes[:, None])
        query_padding_mask, corpus_padding_mask = padding_mask(qgraph_sizes), padding_mask(cgraph_sizes)
        batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
        propagated_features = None

        def flatten_fringe_weights(fringe_weights):
            dummy_query_weights = torch.ones_like(query_padding_mask)
            combined_weights = torch.cat([dummy_query_weights, fringe_weights], dim=-1)
            combined_mask = torch.cat([query_padding_mask, corpus_padding_mask], dim=-1)
            return combined_weights[combined_mask]

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)

        for masking_iter in range(self.masking_iters):
          for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
              node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, flatten_fringe_weights(fringe_weights), edge_features_enc)

          if masking_iter == self.masking_iters - 1:
              propagated_features = torch.clone(node_features_enc)
          node_features_enc = self.fc_transform2(self.relu1(self.fc_transform1(node_features_enc)))

          _, _, transport_plan_intermediate = self._pad_and_apply_sinkhorn(node_features_enc, batch_data_sizes_flat)
          fringe_weights = torch.sum(query_padding_mask[:,:,None] * transport_plan_intermediate, dim=1)
        
        stacked_qnode_emb, stacked_cnode_emb, transport_plan = self._pad_and_apply_sinkhorn(propagated_features, batch_data_sizes_flat)

        if self.diagnostic_mode:
            return transport_plan
        
        scores = -torch.sum(torch.maximum(stacked_qnode_emb - transport_plan@stacked_cnode_emb,\
              cudavar(self.av,torch.tensor([0]))),\
           dim=(1,2))

        return scores

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
          print(transport_plan[2])
        if self.diagnostic_mode:
            return transport_plan
        
        scores = -torch.sum(torch.maximum(stacked_qnode_emb - transport_plan@stacked_cnode_emb,\
              cudavar(self.av,torch.tensor([0]))),\
          dim=(1,2))
        
        return scores
