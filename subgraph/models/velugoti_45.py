import torch
import torch.nn.functional as F
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen
from subgraph.utils import cudavar

class OurMatchingModelVar45_GMN_encoding_NodeAndEdgePerm_SinkhornParamBig_HingeScore(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(OurMatchingModelVar45_GMN_encoding_NodeAndEdgePerm_SinkhornParamBig_HingeScore, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
        #---------------------------- Diagonistic mode return change --------------------------------------------
        self.align_mode = "node_aign"

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

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)


        #STEP1: Get node embeddings
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for _ in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)


        #STEP2: Get transport plan on nodes
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
        transport_plan_node = pytorch_sinkhorn_iters(self.av,sinkhorn_input_node)


        #STEP3: Get transport plan on edges
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

        stacked_qedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size_edge-x.shape[0])) \
                                         for x in edge_feature_enc_query])
        stacked_cedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size_edge-x.shape[0])) \
                                         for x in edge_feature_enc_corpus])

        transformed_qedge_emb = self.edge_fc_transform2(self.edge_relu1(self.edge_fc_transform1(stacked_qedge_emb)))
        transformed_cedge_emb = self.edge_fc_transform2(self.edge_relu1(self.edge_fc_transform1(stacked_cedge_emb)))
        qgraph_edge_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map_edge[i] for i in qgraph_edge_sizes]))
        cgraph_edge_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map_edge[i] for i in cgraph_edge_sizes]))
        masked_qedge_emb = torch.mul(qgraph_edge_mask,transformed_qedge_emb)
        masked_cedge_emb = torch.mul(cgraph_edge_mask,transformed_cedge_emb)

        sinkhorn_input_edge = torch.matmul(masked_qedge_emb,masked_cedge_emb.permute(0,2,1))
        transport_plan_edge = pytorch_sinkhorn_iters(self.av,sinkhorn_input_edge)


        #STEP4: Consistency scoring of node and edge alignment
        edge_count_pairs = list(zip(edge_counts[0::2],edge_counts[1::2]))
        from_idx_split = torch.split(from_idx, edge_counts, dim=0)
        to_idx_split = torch.split(to_idx, edge_counts, dim=0)
        
        prefix_sum_node_counts = [sum(batch_data_sizes_flat[:k])for k in range(len(batch_data_sizes_flat))]
        to_idx_split_relabeled = [x1 - x2 for (x1, x2) in zip(to_idx_split, prefix_sum_node_counts)]
        from_idx_split_relabeled = [x1 - x2 for (x1, x2) in zip(from_idx_split, prefix_sum_node_counts)]

        #=====================STRAIGHT===================================================
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

        #=====================CROSS===================================================
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


        if self.diagnostic_mode:
            return transport_plan_node

        final_score = -torch.sum(torch.maximum(stacked_qedge_emb - transport_plan_edge@stacked_cedge_emb,\
              cudavar(self.av,torch.tensor([0]))),\
           dim=(1,2))

        consistency_loss2 = torch.mul((1-stacked_all_node_map_scores\
                                       -stacked_all_node_map_scores1+torch.mul(stacked_all_node_map_scores,\
                                        stacked_all_node_map_scores1)), transport_plan_edge)
        consistency_loss2 = consistency_loss2 + torch.mul((1-transport_plan_edge), stacked_all_node_map_scores)
        consistency_loss2 = consistency_loss2 + torch.mul((1-transport_plan_edge), stacked_all_node_map_scores1)
        consistency_loss2 = torch.sum(consistency_loss2, dim=(1,2))

        consistency_loss3 = torch.maximum(stacked_all_node_map_scores, stacked_all_node_map_scores1)
        consistency_loss3 = torch.abs(transport_plan_edge - consistency_loss3)
        consistency_loss3 = torch.sum(consistency_loss3, dim=(1,2))

        return (final_score, consistency_loss2, consistency_loss3)