import torch
import GMN.utils as gmnutils
import torch.nn.functional as F
from subgraph.utils import cudavar
from GMN.loss import euclidean_distance
import GMN.graphmatchingnetwork as gmngmn
import GMN.graphembeddingnetwork as gmngen
from subgraph.models.gmn_match_hinge_lrl import CrossAttention

class GMN_match_hinge_scoring(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        super(GMN_match_hinge_scoring, self).__init__()
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
        self.cross_attention_module = CrossAttention(self.av, 'none', prop_config['node_state_dim'], self.max_node_size)
        
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
                                                cross_attention_module=self.cross_attention_module)

        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]

        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size+1-x.shape[0])) \
                                         for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size+1-x.shape[0])) \
                                         for x in node_feature_enc_corpus])

        _, attention_matrix = self.cross_attention_module(node_features_enc, batch_data_sizes_flat)

        scores = -torch.sum(torch.maximum(
            stacked_qnode_emb - attention_matrix[0]@stacked_cnode_emb,
            torch.tensor([0], device=self.device)),
           dim=(1,2))
        
        return scores

class GMN_match_hinge_colbert(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        super(GMN_match_hinge_colbert, self).__init__()
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
        self.cross_attention_module = CrossAttention(self.av, 'none', prop_config['node_state_dim'], self.max_node_size)
        
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
                                                cross_attention_module=self.cross_attention_module)

        graph_vectors = self.aggregator(node_features_enc,graph_idx,2*len(batch_data_sizes) )
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)
        #scores = -euclidean_distance(x,y)
        scores = -torch.sum(torch.nn.ReLU()(x-y),dim=-1)
        return scores

class GMN_match_hinge_scoring_sinkhorn(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        super(GMN_match_hinge_scoring_sinkhorn, self).__init__()
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

        self.max_node_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.cross_attention_module = CrossAttention(self.av, 'none', prop_config['node_state_dim'], self.max_node_size, use_sinkhorn=True)
        
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
                                                cross_attention_module=self.cross_attention_module)

        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]

        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size-x.shape[0])) \
                                         for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size-x.shape[0])) \
                                         for x in node_feature_enc_corpus])

        _, attention_matrix = self.cross_attention_module(node_features_enc, batch_data_sizes_flat)

        scores = -torch.sum(torch.maximum(
            stacked_qnode_emb - attention_matrix[0]@stacked_cnode_emb,
            torch.tensor([0], device=self.device)),
           dim=(1,2))
        
        return scores