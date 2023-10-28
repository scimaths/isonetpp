import torch
import torch.nn.functional as F
from subgraph.utils import cudavar
import GMN.graphembeddingnetwork as gmngen
from subgraph.models.utils import pytorch_sinkhorn_iters

class Node_align_Edge_loss(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(Node_align_Edge_loss, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
        
    def build_masking_utility(self):
        self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        #this mask pattern sets bottom last few rows to 0 based on padding needs
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_set_size+1)]
        # Mask pattern sets top left (k)*(k) square to 1 inside arrays of size n*n. Rest elements are 0
        self.set_size_to_mask_map = [torch.cat((torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([x,self.max_set_size-x])).repeat(x,1),
                             torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([0,self.max_set_size])).repeat(self.max_set_size-x,1)))
                             for x in range(0,self.max_set_size+1)]

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
        
        self.edge_score_fc = torch.nn.Linear(self.prop_layer._message_net[-1].out_features, 1)
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    
    
    def compute_edge_scores_from_node_embeds(self,H, H_sz): 
        """
          H    : (batched) node embedding matrix with each row a node embed
          H_sz : garh sizes of all graphs in the batch
        """
        #we want all pair combination of node embeds
        #repeat and repeat_interleave H and designate either of them as source and the other destination 
        source = torch.repeat_interleave(H,repeats=self.max_set_size,dim =1)
        destination =  H.repeat(1,self.max_set_size,1)
        #each edge feature is [1] 
        edge_emb = cudavar(self.av,torch.ones(source.shape[0],source.shape[1],1))
        #Undirected graphs - hence do both forward and backward concat for each edge 
        forward_batch = torch.cat((source,destination,edge_emb),dim=-1)
        backward_batch = torch.cat((destination,source,edge_emb),dim=-1)
        #use message encoding network from GMN encoding to obtain forward and backward score for each edge
        forward_msg_batch = self.edge_score_fc(self.prop_layer._message_net(forward_batch))
        backward_msg_batch = self.edge_score_fc(self.prop_layer._reverse_message_net(backward_batch))
        #design choice to add forward and backward scores to get total edge score
        bidirectional_msg_batch = torch.cat((forward_msg_batch,backward_msg_batch),dim=-1)
        #note the reshape here to get M matrix
        edge_scores_batch = torch.sum(bidirectional_msg_batch,dim=-1).reshape(-1,self.max_set_size,self.max_set_size)
        #mask the rows and cols denoting edges with dummy node either side
        mask_batch = cudavar(self.av,torch.stack([self.set_size_to_mask_map[i] for i in H_sz]))
        masked_edge_scores_batch = torch.mul(edge_scores_batch,mask_batch)    
        #TODO: NOTE: May need to fill diagonal with 0
        return masked_edge_scores_batch

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
        transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input)
 
        qgr_edge_scores_batch = self.compute_edge_scores_from_node_embeds(stacked_qnode_emb,qgraph_sizes)
        cgr_edge_scores_batch = self.compute_edge_scores_from_node_embeds(stacked_cnode_emb,cgraph_sizes)
       
        qgr_edge_scores_batch_masked = torch.mul(qgr_edge_scores_batch,q_adj)
        cgr_edge_scores_batch_masked = torch.mul(cgr_edge_scores_batch,c_adj)
        if self.diagnostic_mode:
            return transport_plan
        
        scores = -torch.sum(torch.maximum(qgr_edge_scores_batch_masked - transport_plan@cgr_edge_scores_batch_masked@transport_plan.permute(0,2,1),\
               cudavar(self.av,torch.tensor([0]))),\
            dim=(1,2))
        
        return scores
