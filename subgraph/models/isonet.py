import torch
import torch.nn.functional as F
from subgraph.utils import cudavar
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen

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
 
        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum(torch.maximum(stacked_qedge_emb - transport_plan@stacked_cedge_emb,\
              cudavar(self.av,torch.tensor([0]))),\
           dim=(1,2))
        
        return scores

class ISONET_Sym(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        super(ISONET_Sym, self).__init__()
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
 
        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum((stacked_qedge_emb - transport_plan@stacked_cedge_emb) ** 2, dim=(1,2))
        
        return scores
