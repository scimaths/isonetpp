import torch
import torch.nn.functional as F
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen
from subgraph.utils import cudavar

class OurMatchingModelVar39_GMN_encoding_NodePerm_SinkhornParamBig_HingeScore_EdgePermConsistency(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(OurMatchingModelVar39_GMN_encoding_NodePerm_SinkhornParamBig_HingeScore_EdgePermConsistency, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
        self.align_mode = "edge_align"

    def build_masking_utility(self):
        self.max_set_size_node = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.max_set_size_edge = self.av.MAX_EDGES
        #this mask pattern sets bottom last few rows to 0 based on padding needs - mask shape (max_set_size_node,transform_dim)
        self.graph_size_to_mask_map_node = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size_node-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_set_size_node+1)]
        #this mask pattern sets bottom last few rows to 0 based on padding needs- mask shape (max_set_size_edge,transform_dim)
        self.graph_size_to_mask_map_edge = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size_edge-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_set_size_edge+1)]
        #this mask pattern sets bottom last few rows to 0 based on padding needs- mask shape (max_set_size_edge,max_set_size_edge)
        self.inconsistency_score_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.max_set_size_edge), \
        torch.tensor([0]).repeat(self.max_set_size_edge-x,1).repeat(1,self.max_set_size_edge))) for x in range(0,self.max_set_size_edge+1)]



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
        self.node_fc_transform1 = torch.nn.Linear(self.av.filters_3, self.av.transform_dim)
        self.node_relu1 = torch.nn.ReLU()
        self.node_fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)

        #NOTE:FILTERS_3 is 10 for now - hardcoded into config
        self.edge_fc_transform1 = torch.nn.Linear(2*self.av.filters_3, self.av.transform_dim)
        self.edge_relu1 = torch.nn.ReLU()
        self.edge_fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)

        self.score_aggr_ff = torch.nn.Linear(2, 1)


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
        #A
        #a, b = zip(*batch_adj)
        #q_adj = torch.stack(a)
        #c_adj = torch.stack(b)


        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)

        #STEP1: Get node embeddings
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            #node_feature_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)

        #STEP2: Get transport plan on nodes
        #[(8, 12), (10, 13), (10, 14)] -> [8, 12, 10, 13, 10, 14]
        batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]
        assert(list(zip([x.shape[0] for x in node_feature_enc_query], \
                        [x.shape[0] for x in node_feature_enc_corpus])) \
               == batch_data_sizes)


        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size_node-x.shape[0])) \
                                         for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size_node-x.shape[0])) \
                                         for x in node_feature_enc_corpus])


        transformed_qnode_emb = self.node_fc_transform2(self.node_relu1(self.node_fc_transform1(stacked_qnode_emb)))
        transformed_cnode_emb = self.node_fc_transform2(self.node_relu1(self.node_fc_transform1(stacked_cnode_emb)))
        qgraph_node_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map_node[i] for i in qgraph_sizes]))
        cgraph_node_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map_node[i] for i in cgraph_sizes]))
        masked_qnode_emb = torch.mul(qgraph_node_mask,transformed_qnode_emb)
        masked_cnode_emb = torch.mul(cgraph_node_mask,transformed_cnode_emb)

        sinkhorn_input_node = torch.matmul(masked_qnode_emb,masked_cnode_emb.permute(0,2,1))
        if self.av.transport_node_type == "soft":
            transport_plan_node = pytorch_sinkhorn_iters(self.av,sinkhorn_input_node)
        elif self.av.transport_node_type == "hard":
            transport_plan_node_soft = pytorch_sinkhorn_iters(self.av,sinkhorn_input_node)
        
            transport_plan_node_soft_cpu = [x.detach().cpu().numpy() for x in transport_plan_node_soft]
            transport_plan_node = []
            for plan in transport_plan_node_soft_cpu:
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(-plan)
                transport_plan_node.append(torch.eye(self.max_set_size_node)[col_ind])
            transport_plan_node = cudavar(self.av,torch.stack(transport_plan_node))
        

        #STEP3: Get edge embeddings from node embeddings
        source_node_enc = node_features_enc[from_idx]
        dest_node_enc  = node_features_enc[to_idx]
        forward_edge_input = torch.cat((source_node_enc,dest_node_enc,edge_features_enc),dim=-1)
        backward_edge_input = torch.cat((dest_node_enc,source_node_enc,edge_features_enc),dim=-1)
        forward_edge_msg = self.prop_layer._message_net(forward_edge_input)
        backward_edge_msg = self.prop_layer._reverse_message_net(backward_edge_input)
        edge_msg_enc = forward_edge_msg + backward_edge_msg

        edge_counts  = self.fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
        qgraph_edge_sizes = cudavar(self.av,torch.tensor(edge_counts[0::2]))
        cgraph_edge_sizes = cudavar(self.av,torch.tensor(edge_counts[1::2]))

        edge_feature_enc_split = torch.split(edge_msg_enc, edge_counts, dim=0)
        edge_feature_enc_query = edge_feature_enc_split[0::2]
        edge_feature_enc_corpus = edge_feature_enc_split[1::2]

        #STEP4: Get transport plan on edges

        stacked_qedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size_edge-x.shape[0])) \
                                         for x in edge_feature_enc_query])
        stacked_cedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size_edge-x.shape[0])) \
                                         for x in edge_feature_enc_corpus])
        #STEP5: Consistency scoring of node and edge alignment
        edge_count_pairs = list(zip(edge_counts[0::2],edge_counts[1::2]))
        from_idx_split = torch.split(from_idx, edge_counts, dim=0)
        to_idx_split = torch.split(to_idx, edge_counts, dim=0)
        #Not most efficient but doesn't matter for such small lengths
        #Input :[10, 15, 10, 14, 10, 15, 9, 13]
        #Output:[0, 10, 25, 35, 49, 59, 74, 83]
        prefix_sum_node_counts = [sum(batch_data_sizes_flat[:k])for k in range(len(batch_data_sizes_flat))]
        to_idx_split_relabeled = [x1 - x2 for (x1, x2) in zip(to_idx_split, prefix_sum_node_counts)]
        from_idx_split_relabeled = [x1 - x2 for (x1, x2) in zip(from_idx_split, prefix_sum_node_counts)]


        #=================STRAIGHT=============
        from_node_ids_mapping = [torch.cartesian_prod(x,y) \
           for (x,y) in zip(from_idx_split_relabeled[0::2], from_idx_split_relabeled[1::2])]
        from_node_map_scores = [x[y[:,0],y[:,1]].view(z) for (x,y,z) in zip(transport_plan_node,from_node_ids_mapping,edge_count_pairs)]
        to_node_ids_mapping = [torch.cartesian_prod(x,y) \
           for (x,y) in zip(to_idx_split_relabeled[0::2], to_idx_split_relabeled[1::2])]
        to_node_map_scores = [x[y[:,0],y[:,1]].view(z) for (x,y,z) in zip(transport_plan_node,to_node_ids_mapping,edge_count_pairs)]
        stacked_from_node_map_scores = torch.stack([F.pad(x, pad=(0,self.max_set_size_edge-x.shape[1],0,self.max_set_size_edge-x.shape[0])) \
                                 for x in from_node_map_scores])
        stacked_to_node_map_scores = torch.stack([F.pad(x, pad=(0,self.max_set_size_edge-x.shape[1],0,self.max_set_size_edge-x.shape[0])) \
                                 for x in to_node_map_scores])
        stacked_all_node_map_scores = torch.mul(stacked_from_node_map_scores,stacked_to_node_map_scores)
        
        

        #==================CROSS=========
        from_node_ids_mapping1 = [torch.cartesian_prod(x,y) \
           for (x,y) in zip(from_idx_split_relabeled[0::2], to_idx_split_relabeled[1::2])]
        from_node_map_scores1 = [x[y[:,0],y[:,1]].view(z) for (x,y,z) in zip(transport_plan_node,from_node_ids_mapping1,edge_count_pairs)]
        to_node_ids_mapping1 = [torch.cartesian_prod(x,y) \
           for (x,y) in zip(to_idx_split_relabeled[0::2], from_idx_split_relabeled[1::2])]
        to_node_map_scores1 = [x[y[:,0],y[:,1]].view(z) for (x,y,z) in zip(transport_plan_node,to_node_ids_mapping1,edge_count_pairs)]
        stacked_from_node_map_scores1 = torch.stack([F.pad(x, pad=(0,self.max_set_size_edge-x.shape[1],0,self.max_set_size_edge-x.shape[0])) \
                                 for x in from_node_map_scores1])
        stacked_to_node_map_scores1 = torch.stack([F.pad(x, pad=(0,self.max_set_size_edge-x.shape[1],0,self.max_set_size_edge-x.shape[0])) \
                                 for x in to_node_map_scores1])
        stacked_all_node_map_scores1 = torch.mul(stacked_from_node_map_scores1,stacked_to_node_map_scores1)
        
    
        transport_plan_edge = torch.add(stacked_all_node_map_scores, stacked_all_node_map_scores1)
    

        if self.av.transport_edge_type == "sinkhorn":
            transport_plan_edge = pytorch_sinkhorn_iters(self.av,transport_plan_edge)
    

        if self.diagnostic_mode:
            if self.align_mode == "edge_align":
                return transport_plan_edge
            elif self.align_mode == "node_align":
                return transport_plan_node
        
        scores_node_align = -torch.sum(torch.maximum(stacked_qnode_emb - transport_plan_node@stacked_cnode_emb,\
             cudavar(self.av,torch.tensor([0]))),\
           dim=(1,2))
        scores_kronecker_edge_align = cudavar(self.av,torch.tensor(self.av.consistency_lambda)) * (-torch.sum(torch.maximum(stacked_qedge_emb - transport_plan_edge@stacked_cedge_emb,\
cudavar(self.av,torch.tensor([0]))),dim=(1,2)))
        
        final_score = 0
        final_score += scores_node_align
        final_score += scores_kronecker_edge_align                
        return final_score
        
        #return scores_node_align + (cudavar(self.av,torch.tensor(self.av.IPLUS_LAMBDA)) * \
        #                            regularizer_consistency)
    
    