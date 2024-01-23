import torch
import GMN.utils as gmnutils
import torch.nn.functional as F
from subgraph.utils import cudavar
from GMN.loss import euclidean_distance
import GMN.graphembeddingnetwork as gmngen
from subgraph.models.utils import colbert_scores_for_gmn_data

class GMN_embed(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(GMN_embed, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        
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
        self.aggregator = gmngen.GraphAggregator(**self.config['aggregator'])
        
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
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            
        graph_vectors = self.aggregator(node_features_enc,graph_idx,2*len(batch_data_sizes) )
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)
        scores = -euclidean_distance(x,y)
        
        return scores

class GMN_embed_with_ColBERT_scores(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(GMN_embed_with_ColBERT_scores, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        
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
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            
        return colbert_scores_for_gmn_data(node_features_enc, batch_data_sizes, graph_idx)

class GMN_embed_hinge(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        super(GMN_embed_hinge, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_layers()
        
    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)      
        self.aggregator = gmngen.GraphAggregator(**self.config['aggregator'])
        
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
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            #node_feature_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            
        graph_vectors = self.aggregator(node_features_enc,graph_idx,2*len(batch_data_sizes) )
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)
        #scores = -euclidean_distance(x,y)
        scores = -torch.sum(torch.nn.ReLU()(x-y),dim=-1)
        return scores

class GMN_embed_hinge_scoring(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        super(GMN_embed_hinge_scoring, self).__init__()
        self.av = av
        self.device = 'cuda:0' if self.av.has_cuda and self.av.want_cuda else 'cpu'
        self.config = config
        self.input_dim = input_dim
        self.build_layers()

    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)      
        self.aggregator = gmngen.GraphAggregator(**self.config['aggregator'])

        self.max_node_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.cross_attention_module = CrossAttention(self.av, 'lrl', prop_config['node_state_dim'], self.max_node_size)

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
        """
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)

        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]

        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size-x.shape[0])) \
                                         for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size-x.shape[0])) \
                                         for x in node_feature_enc_corpus])

        _, transport_plan = self.cross_attention_module(node_features_enc, batch_data_sizes_flat)

        scores = -torch.sum(torch.maximum(
            stacked_qnode_emb - transport_plan@stacked_cnode_emb,
            torch.tensor([0], device=self.device)),
           dim=(1,2))

        return scores

class GMN_embed_with_MLP_and_ColBERT_scores(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(GMN_embed_with_MLP_and_ColBERT_scores, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        
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
        agg_config = self.config['aggregator']
        self.node_feature_processor = gmngen.NodeFeatureProcessor(agg_config['node_hidden_sizes'], agg_config['input_size'])

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
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)

        node_features_enc = self.node_feature_processor(node_features_enc)

        return colbert_scores_for_gmn_data(node_features_enc, batch_data_sizes, graph_idx)


class GMN_embed_maxsim_dot(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(GMN_embed_maxsim_dot, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.build_layers()

    def build_layers(self):

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)      
        self.aggregator = gmngen.GraphAggregator(**self.config['aggregator'])
        
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
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            #node_feature_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)

        node_features_enc = self.aggregator.MLP1(node_features_enc)
        if self.aggregator._gated:
            gates = torch.sigmoid(node_features_enc[:, :self.aggregator._graph_state_dim])
            node_features_enc = node_features_enc[:, self.aggregator._graph_state_dim:] * gates

        node_counts = torch.tensor(batch_data_sizes, device=node_features_enc.device).flatten()
        node_feature_enc_split = torch.split(node_features_enc, node_counts.tolist(), dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]
        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                            for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                            for x in node_feature_enc_corpus])

        #scores = -euclidean_distance(x,y)
        # scores = -torch.sum(torch.nn.ReLU()(x-y),dim=-1)
        scores = (((stacked_qnode_emb[:,:,None,:]*stacked_cnode_emb[:,None,:,:]).sum(-1)).max(-1).values).sum(-1)

        return scores

class GMN_embed_maxsim_dot_corrected(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(GMN_embed_maxsim_dot_corrected, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.build_layers()

    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)      
        self.aggregator = gmngen.GraphAggregator(**self.config['aggregator'])
        
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
        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            #node_feature_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)

        node_features_enc = self.aggregator.MLP1(node_features_enc)
        if self.aggregator._gated:
            gates = torch.sigmoid(node_features_enc[:, :self.aggregator._graph_state_dim])
            node_features_enc = node_features_enc[:, self.aggregator._graph_state_dim:] * gates

        node_counts = torch.tensor(batch_data_sizes, device=node_features_enc.device)
        node_feature_enc_split = torch.split(node_features_enc, node_counts.flatten().tolist(), dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]
        stacked_qnode_emb = torch.stack([torch.nn.functional.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                            for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([torch.nn.functional.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                            for x in node_feature_enc_corpus])
        scores = (stacked_qnode_emb[:,:,None,:]*stacked_cnode_emb[:,None,:,:]).sum(-1)
        
        def get_padding_mask(node_counts):
            return torch.arange(0, self.max_set_size, device=node_counts.device)[None,:] < node_counts[:,None]

        combined_padding_mask = get_padding_mask(node_counts[:, 0])[:,:,None] * ~get_padding_mask(node_counts[:, 1])[:,None,:]
        scores.masked_fill_(combined_padding_mask, -torch.inf)
        return scores.max(-1).values.sum(-1)
