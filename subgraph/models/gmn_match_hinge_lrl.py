import torch
import GMN.utils as gmnutils
import torch.nn.functional as F
from subgraph.utils import cudavar
from GMN.loss import euclidean_distance
import GMN.graphmatchingnetwork as gmngmn
import GMN.graphembeddingnetwork as gmngen
from subgraph.models.utils import pytorch_sinkhorn_iters

class CrossAttention(torch.nn.Module):
    def __init__(self, av, type_, dim, max_node_size, use_sinkhorn=False, attention_to_sinkhorn=False):
        super(CrossAttention, self).__init__()
        self.av = av
        self.device = 'cuda:0' if self.av.has_cuda and self.av.want_cuda else 'cpu'
        self.type = type_
        self.dim = dim
        self.use_sinkhorn = use_sinkhorn
        self.attention_to_sinkhorn = attention_to_sinkhorn
        self.max_node_size = max_node_size
        if not self.use_sinkhorn:
            self.max_node_size = self.max_node_size + 1
        self.build_layers()
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1], device=self.device).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0], device=self.device).repeat(self.max_node_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_node_size+1)]
        
    def build_layers(self):
        if self.type == 'lrl':
            self.fc_transform1 = torch.nn.Linear(self.dim, self.av.transform_dim)
            self.relu1 = torch.nn.ReLU()
            self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        elif self.type == 'hinge':
            self.relu1 = torch.nn.ReLU()

    def forward(self, data, batch_data_sizes_flat, return_dotpdt=False):

        partitionsT = torch.split(data, batch_data_sizes_flat)
        partitions_1 = torch.stack([F.pad(partition, pad=(0, 0, 0, self.max_node_size-len(partition))) for partition in partitionsT[0::2]])
        partitions_2 = torch.stack([F.pad(partition, pad=(0, 0, 0, self.max_node_size-len(partition))) for partition in partitionsT[1::2]])

        # mask
        mask_11 = torch.stack([F.pad(torch.ones_like(partition), pad=(0, 0, 0, self.max_node_size-len(partition))) for partition in partitionsT[0::2]])
        mask_12 = torch.stack([F.pad(torch.zeros_like(partition), pad=(0, 0, 0, self.max_node_size-len(partition)), value=1) for partition in partitionsT[0::2]])
        mask_21 = torch.stack([F.pad(torch.ones_like(partition), pad=(0, 0, 0, self.max_node_size-len(partition))) for partition in partitionsT[1::2]])
        mask_22 = torch.stack([F.pad(torch.zeros_like(partition), pad=(0, 0, 0, self.max_node_size-len(partition)), value=1) for partition in partitionsT[1::2]])

        mask = torch.bmm(mask_11, torch.transpose(mask_21, 1, 2))
        mask += torch.bmm(mask_12, torch.transpose(mask_22, 1, 2))
        mask = (1 - (mask//data.shape[1])).to(dtype=torch.bool)

        # transform
        if self.type == 'lrl':
            transformed_partitions_1 = self.fc_transform2(self.relu1(self.fc_transform1(partitions_1)))
            transformed_partitions_2 = self.fc_transform2(self.relu1(self.fc_transform1(partitions_2)))
            transform_mask_1 = torch.stack([self.graph_size_to_mask_map[i] for i in batch_data_sizes_flat[0::2]])
            transform_mask_2 = torch.stack([self.graph_size_to_mask_map[i] for i in batch_data_sizes_flat[1::2]])
            transformed_partitions_1 = torch.mul(transform_mask_1, transformed_partitions_1)
            transformed_partitions_2 = torch.mul(transform_mask_2, transformed_partitions_2)
            dot_pdt_similarity = torch.bmm(transformed_partitions_1, torch.transpose(transformed_partitions_2, 1, 2))
        elif self.type == 'hinge':
            batch_size = partitions_1.shape[0]
            transformed_partitions_1 = partitions_1.repeat(1, 1, self.max_node_size).reshape(batch_size, self.max_node_size*self.max_node_size, data.shape[1])
            transformed_partitions_2 = partitions_2.repeat(1, self.max_node_size, 1)
            dot_pdt_similarity = torch.sum(-self.relu1(transformed_partitions_1 - transformed_partitions_2), dim=2).reshape(batch_size, self.max_node_size, self.max_node_size)
        else:
            dot_pdt_similarity = torch.bmm(partitions_1, torch.transpose(partitions_2, 1, 2))

        if self.use_sinkhorn:
            # transport plan
            transport_plan = pytorch_sinkhorn_iters(self.av, dot_pdt_similarity)
            
            # final
            query_new = torch.bmm(transport_plan, partitions_2)
            corpus_new = torch.bmm(torch.transpose(transport_plan, 1, 2), partitions_1)
            
            results = torch.cat([query_new[i//2, :batch_data_sizes_flat[i]] if i%2==0 else corpus_new[i//2, :batch_data_sizes_flat[i]] for i in range(len(batch_data_sizes_flat))])
            
            return results, transport_plan
        else:
            # softmax
            # mask to fill -inf
            if not self.attention_to_sinkhorn:
                dot_pdt_similarity.masked_fill_(mask, -torch.inf)
            if return_dotpdt:
                return dot_pdt_similarity
            dot_pdt_similarity = torch.div(dot_pdt_similarity, self.av.temp_gmn_scoring)
            softmax_1 = torch.softmax(dot_pdt_similarity, dim=2)
            softmax_2 = torch.softmax(dot_pdt_similarity, dim=1)

            # final
            query_new = torch.bmm(softmax_1, partitions_2)
            corpus_new = torch.bmm(torch.transpose(softmax_2, 1, 2), partitions_1)

            results = torch.cat([query_new[i//2, :batch_data_sizes_flat[i]] if i%2==0 else corpus_new[i//2, :batch_data_sizes_flat[i]] for i in range(len(batch_data_sizes_flat))])
            
            return results, [softmax_1, softmax_2]


class GMN_match_hinge_lrl(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        super(GMN_match_hinge_lrl, self).__init__()
        self.av = av
        self.device = 'cuda:0' if self.av.has_cuda and self.av.want_cuda else 'cpu'
        self.config = config
        self.input_dim = input_dim
        self.build_layers()
        
    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        self.similarity_func = self.config['graph_matching_net']['similarity']
        prop_config = self.config['graph_matching_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        prop_config.pop('similarity',None)        
        self.prop_layer = gmngmn.GraphPropMatchingLayer(**prop_config)      
        self.aggregator = gmngen.GraphAggregator(**self.config['aggregator'])

        self.max_node_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.lrl_cross_attention_module = CrossAttention(self.av, 'lrl', prop_config['node_state_dim'], self.max_node_size)
        
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_node_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_node_size+1)]

    def get_graph(self, batch):
        graph = batch
        node_features = torch.tensor(graph.node_features, device=self.device)
        edge_features = torch.tensor(graph.edge_features, device=self.device)
        from_idx = torch.tensor(graph.from_idx, dtype=torch.int64, device=self.device)
        to_idx = torch.tensor(graph.to_idx, dtype=torch.int64, device=self.device)
        graph_idx = torch.tensor(graph.graph_idx, dtype=torch.int64, device=self.device)
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
          batch_adj is unused
        """
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_matching_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,\
                                                graph_idx,2*len(batch_data_sizes), \
                                                self.similarity_func, edge_features_enc, 
                                                batch_data_sizes_flat=batch_data_sizes_flat, 
                                                max_node_size=self.max_node_size,
                                                cross_attention_module=self.lrl_cross_attention_module)

        graph_vectors = self.aggregator(node_features_enc,graph_idx,2*len(batch_data_sizes) )
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)
        scores = -torch.sum(torch.nn.ReLU()(x-y),dim=-1)
        return scores

class GMN_match_hinge_lrl_scoring(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        super(GMN_match_hinge_lrl_scoring, self).__init__()
        self.av = av
        self.device = 'cuda:0' if self.av.has_cuda and self.av.want_cuda else 'cpu'
        self.config = config
        self.input_dim = input_dim
        self.build_layers()
        
    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        self.similarity_func = self.config['graph_matching_net']['similarity']
        prop_config = self.config['graph_matching_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        prop_config.pop('similarity',None)        
        self.prop_layer = gmngmn.GraphPropMatchingLayer(**prop_config)      
        self.aggregator = gmngen.GraphAggregator(**self.config['aggregator'])

        self.max_node_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.lrl_cross_attention_module = CrossAttention(self.av, 'lrl', prop_config['node_state_dim'], self.max_node_size)
        
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_node_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_node_size+1)]

    def get_graph(self, batch):
        graph = batch
        node_features = torch.tensor(graph.node_features, device=self.device)
        edge_features = torch.tensor(graph.edge_features, device=self.device)
        from_idx = torch.tensor(graph.from_idx, dtype=torch.int64, device=self.device)
        to_idx = torch.tensor(graph.to_idx, dtype=torch.int64, device=self.device)
        graph_idx = torch.tensor(graph.graph_idx, dtype=torch.int64, device=self.device)
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
          batch_adj is unused
        """
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_matching_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,\
                                                graph_idx,2*len(batch_data_sizes), \
                                                self.similarity_func, edge_features_enc, 
                                                batch_data_sizes_flat=batch_data_sizes_flat, 
                                                max_node_size=self.max_node_size,
                                                cross_attention_module=self.lrl_cross_attention_module)

        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]

        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size+1-x.shape[0])) \
                                         for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size+1-x.shape[0])) \
                                         for x in node_feature_enc_corpus])

        _, attention_matrix = self.lrl_cross_attention_module(node_features_enc, batch_data_sizes_flat)

        scores = -torch.sum(torch.maximum(
            stacked_qnode_emb - attention_matrix[0]@stacked_cnode_emb,
            torch.tensor([0], device=self.device)),
           dim=(1,2))
        
        return scores

class GMN_match_hinge_hinge_similarity(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        super(GMN_match_hinge_hinge_similarity, self).__init__()
        self.av = av
        self.device = 'cuda:0' if self.av.has_cuda and self.av.want_cuda else 'cpu'
        self.config = config
        self.input_dim = input_dim
        self.build_layers()
        
    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        self.similarity_func = self.config['graph_matching_net']['similarity']
        prop_config = self.config['graph_matching_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        prop_config.pop('similarity',None)        
        self.prop_layer = gmngmn.GraphPropMatchingLayer(**prop_config)      
        self.aggregator = gmngen.GraphAggregator(**self.config['aggregator'])

        self.max_node_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.hinge_cross_attention_module = CrossAttention(self.av, 'hinge', prop_config['node_state_dim'], self.max_node_size)
        
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_node_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_node_size+1)]

    def get_graph(self, batch):
        graph = batch
        node_features = torch.tensor(graph.node_features, device=self.device)
        edge_features = torch.tensor(graph.edge_features, device=self.device)
        from_idx = torch.tensor(graph.from_idx, dtype=torch.int64, device=self.device)
        to_idx = torch.tensor(graph.to_idx, dtype=torch.int64, device=self.device)
        graph_idx = torch.tensor(graph.graph_idx, dtype=torch.int64, device=self.device)
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
          batch_adj is unused
        """
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_matching_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,\
                                                graph_idx,2*len(batch_data_sizes), \
                                                self.similarity_func, edge_features_enc, 
                                                batch_data_sizes_flat=batch_data_sizes_flat, 
                                                max_node_size=self.max_node_size,
                                                cross_attention_module=self.hinge_cross_attention_module)

        graph_vectors = self.aggregator(node_features_enc,graph_idx,2*len(batch_data_sizes) )
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)
        scores = -torch.sum(torch.nn.ReLU()(x-y),dim=-1)
        return scores

class GMN_match_hinge_hinge_similarity_scoring(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        super(GMN_match_hinge_hinge_similarity_scoring, self).__init__()
        self.av = av
        self.device = 'cuda:0' if self.av.has_cuda and self.av.want_cuda else 'cpu'
        self.config = config
        self.input_dim = input_dim
        self.build_layers()
        
    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        self.similarity_func = self.config['graph_matching_net']['similarity']
        prop_config = self.config['graph_matching_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        prop_config.pop('similarity',None)        
        self.prop_layer = gmngmn.GraphPropMatchingLayer(**prop_config)      
        self.aggregator = gmngen.GraphAggregator(**self.config['aggregator'])

        self.max_node_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.hinge_cross_attention_module = CrossAttention(self.av, 'hinge', prop_config['node_state_dim'], self.max_node_size)
        
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_node_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_node_size+1)]

    def get_graph(self, batch):
        graph = batch
        node_features = torch.tensor(graph.node_features, device=self.device)
        edge_features = torch.tensor(graph.edge_features, device=self.device)
        from_idx = torch.tensor(graph.from_idx, dtype=torch.int64, device=self.device)
        to_idx = torch.tensor(graph.to_idx, dtype=torch.int64, device=self.device)
        graph_idx = torch.tensor(graph.graph_idx, dtype=torch.int64, device=self.device)
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
          batch_adj is unused
        """
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_matching_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,\
                                                graph_idx,2*len(batch_data_sizes), \
                                                self.similarity_func, edge_features_enc, 
                                                batch_data_sizes_flat=batch_data_sizes_flat, 
                                                max_node_size=self.max_node_size,
                                                cross_attention_module=self.hinge_cross_attention_module)

        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]

        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size+1-x.shape[0])) \
                                         for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size+1-x.shape[0])) \
                                         for x in node_feature_enc_corpus])

        _, attention_matrix = self.hinge_cross_attention_module(node_features_enc, batch_data_sizes_flat)

        scores = -torch.sum(torch.maximum(
            stacked_qnode_emb - attention_matrix[0]@stacked_cnode_emb,
            torch.tensor([0], device=self.device)),
           dim=(1,2))
        return scores