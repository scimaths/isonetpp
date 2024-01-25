import torch
import GMN.utils as gmnutils
import torch.nn.functional as F
from subgraph.utils import cudavar
from GMN.loss import euclidean_distance
import GMN.graphmatchingnetwork as gmngmn
import GMN.graphembeddingnetwork as gmngen
from subgraph.models.utils import pytorch_sinkhorn_iters
from subgraph.models.gmn_match_hinge_lrl import CrossAttention

class GMN_match_hinge_vaibhav(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        super(GMN_match_hinge_vaibhav, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.device = 'cuda:0' if self.av.has_cuda and self.av.want_cuda else 'cpu'
        self.build_layers()
        
    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        self.similarity_func = self.config['graph_matching_net']['similarity']
        prop_config = self.config['graph_matching_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        prop_config.pop('similarity',None)        
        self.prop_layer = gmngmn.GraphPropMatchingLayer(**prop_config)      
        self.max_node_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE) + 1

        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1], device=self.device).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0], device=self.device).repeat(self.max_node_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_node_size+1)]

        self.fc_transform1 = torch.nn.Linear(self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)

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
        qgraph_sizes, cgraph_sizes = zip(*batch_data_sizes)
        qgraph_sizes = torch.tensor(qgraph_sizes, device=self.device)
        cgraph_sizes = torch.tensor(cgraph_sizes, device=self.device)

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_matching_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,\
                                                graph_idx,2*len(batch_data_sizes), \
                                                self.similarity_func, edge_features_enc, 
                                                batch_data_sizes_flat=batch_data_sizes_flat, 
                                                max_node_size=self.max_node_size - 1)
            
        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]       

        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size-x.shape[0])) \
                                         for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size-x.shape[0])) \
                                         for x in node_feature_enc_corpus])

        transformed_qnode_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qnode_emb)))
        transformed_cnode_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cnode_emb)))

        qgraph_mask = torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_sizes])
        cgraph_mask = torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_sizes])
        masked_qnode_emb = torch.mul(qgraph_mask,transformed_qnode_emb)
        masked_cnode_emb = torch.mul(cgraph_mask,transformed_cnode_emb)

        sinkhorn_input = torch.matmul(masked_qnode_emb, masked_cnode_emb.permute(0,2,1))
        transport_plan = pytorch_sinkhorn_iters(self.av, sinkhorn_input)

        scores = -torch.sum(torch.maximum(
            stacked_qnode_emb - transport_plan@stacked_cnode_emb,
            torch.tensor([0], device=self.device)),
           dim=(1,2))

        return scores

class GMN_match_hinge_vaibhav_injective_attention(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        super(GMN_match_hinge_vaibhav_injective_attention, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.device = 'cuda:0' if self.av.has_cuda and self.av.want_cuda else 'cpu'
        self.build_layers()
        
    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        self.similarity_func = self.config['graph_matching_net']['similarity']
        prop_config = self.config['graph_matching_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        prop_config.pop('similarity',None)        
        self.prop_layer = gmngmn.GraphPropMatchingLayer(**prop_config)      
        self.max_node_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE) + 1
        self.lrl_cross_attention_module = CrossAttention(self.av, 'lrl', prop_config['node_state_dim'], self.max_node_size, injective_attention=True)

        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1], device=self.device).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0], device=self.device).repeat(self.max_node_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_node_size+1)]

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
        qgraph_sizes, cgraph_sizes = zip(*batch_data_sizes)
        qgraph_sizes = torch.tensor(qgraph_sizes, device=self.device)
        cgraph_sizes = torch.tensor(cgraph_sizes, device=self.device)

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_matching_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,\
                                                graph_idx,2*len(batch_data_sizes), \
                                                self.similarity_func, edge_features_enc, 
                                                batch_data_sizes_flat=batch_data_sizes_flat, 
                                                max_node_size=self.max_node_size - 1)
            
        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]       

        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size-x.shape[0])) \
                                         for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_size-x.shape[0])) \
                                         for x in node_feature_enc_corpus])

        _, attention_matrix = self.lrl_cross_attention_module(node_features_enc, batch_data_sizes_flat)

        scores = -torch.sum(torch.maximum(
            stacked_qnode_emb - attention_matrix[0]@stacked_cnode_emb,
            torch.tensor([0], device=self.device)),
           dim=(1,2))

        return scores