import torch
import torch.nn.functional as F
from subgraph.utils import cudavar
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen

class NodeEarlyInteractionWithConsistency(torch.nn.Module):
    def __init__(self, av, config, input_dim):
        super(NodeEarlyInteractionWithConsistency, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
    
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

    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        if self.config['early_interaction']['prop_separate_params']:
            self.prop_layers = torch.nn.ModuleList([gmngen.GraphPropLayer(**prop_config) for _ in range(self.config['early_interaction']['n_time_updates'])])
        else:
            self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        
        self.fc_combine_interaction = torch.nn.Sequential(
            torch.nn.Linear(2 * prop_config['node_state_dim'], 2 * prop_config['node_state_dim']),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * prop_config['node_state_dim'], prop_config['node_state_dim'])
        )
        self.fc_transform1 = torch.nn.Linear(self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)

    def forward(self, batch_data, batch_data_sizes, batch_adj):
        qgraph_sizes, cgraph_sizes = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av, torch.tensor(qgraph_sizes))
        device = qgraph_sizes.device
        cgraph_sizes = torch.tensor(cgraph_sizes, device=device)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]
        batch_data_sizes_flat_tensor = torch.tensor(batch_data_sizes_flat, device=device, dtype=torch.long)
        cumulative_sizes = torch.cumsum(torch.tensor(self.max_set_size_node, dtype=torch.long, device=device).repeat(len(batch_data_sizes_flat_tensor)), dim=0)

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
        
        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        num_nodes, node_feature_dim = encoded_node_features.shape
        batch_size = len(batch_data_sizes)

        n_time_update_steps = self.config['early_interaction']['n_time_updates']
        n_prop_update_steps = self.config['graph_embedding_net']['n_prop_layers']

        node_feature_store = torch.zeros(num_nodes, node_feature_dim * (n_prop_update_steps + 1), device=node_features.device)
        updated_node_feature_store = torch.zeros_like(node_feature_store)

        max_set_size_arange = torch.arange(self.max_set_size_node, dtype=torch.long, device=device).reshape(1, -1).repeat(batch_size * 2, 1)
        node_presence_mask = max_set_size_arange < batch_data_sizes_flat_tensor.unsqueeze(1)
        max_set_size_arange[1:, ] += cumulative_sizes[:-1].unsqueeze(1)
        node_indices = max_set_size_arange[node_presence_mask]

        for time_idx in range(1, n_time_update_steps + 1):
            node_features_enc = torch.clone(encoded_node_features)
            edge_features_enc = torch.clone(encoded_edge_features)
            for prop_idx in range(1, n_prop_update_steps + 1) :
                nf_idx = node_feature_dim * prop_idx
                if self.config['early_interaction']['time_update_idx'] == "k_t":
                    interaction_features = node_feature_store[:, nf_idx - node_feature_dim : nf_idx]
                elif self.config['early_interaction']['time_update_idx'] == "kp1_t":
                    interaction_features = node_feature_store[:, nf_idx : nf_idx + node_feature_dim]
                
                combined_features = self.fc_combine_interaction(torch.cat([node_features_enc, interaction_features], dim=1))
                if self.config['early_interaction']['prop_separate_params']:
                    node_features_enc = self.prop_layers[time_idx - 1](combined_features, from_idx, to_idx, edge_features_enc)
                else:
                    node_features_enc = self.prop_layer(combined_features, from_idx, to_idx, edge_features_enc)
                updated_node_feature_store[:, nf_idx : nf_idx + node_feature_dim] = torch.clone(node_features_enc)

            node_feature_store = torch.clone(updated_node_feature_store)

            node_feature_store_split = torch.split(node_feature_store, batch_data_sizes_flat, dim=0)
            node_feature_store_query = node_feature_store_split[0::2]
            node_feature_store_corpus = node_feature_store_split[1::2]

            stacked_qnode_store_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size_node-x.shape[0])) for x in node_feature_store_query])
            stacked_cnode_store_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size_node-x.shape[0])) for x in node_feature_store_corpus])
            
            # Compute transport plan
            stacked_qnode_final_emb = stacked_qnode_store_emb[:,:,-node_feature_dim:]
            stacked_cnode_final_emb = stacked_cnode_store_emb[:,:,-node_feature_dim:]
            transformed_qnode_final_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qnode_final_emb)))
            transformed_cnode_final_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cnode_final_emb)))
            
            qgraph_mask = cudavar(self.av, torch.stack([self.graph_size_to_mask_map_node[i] for i in qgraph_sizes]))
            cgraph_mask = cudavar(self.av, torch.stack([self.graph_size_to_mask_map_node[i] for i in cgraph_sizes]))
            masked_qnode_final_emb = torch.mul(qgraph_mask,transformed_qnode_final_emb)
            masked_cnode_final_emb = torch.mul(cgraph_mask,transformed_cnode_final_emb)

            sinkhorn_input = torch.matmul(masked_qnode_final_emb, masked_cnode_final_emb.permute(0, 2, 1))
            transport_plan_node = pytorch_sinkhorn_iters(self.av, sinkhorn_input)

            # Compute interaction
            qnode_features_from_cnodes = torch.bmm(transport_plan_node, stacked_cnode_store_emb)
            cnode_features_from_qnodes = torch.bmm(transport_plan_node.permute(0, 2, 1), stacked_qnode_store_emb)
            interleaved_node_features = torch.cat([
                qnode_features_from_cnodes.unsqueeze(1),
                cnode_features_from_qnodes.unsqueeze(1)
            ], dim=1)[:, :, :, node_feature_dim:].reshape(-1, n_prop_update_steps * node_feature_dim) 
            node_feature_store[:, node_feature_dim:] = interleaved_node_features[node_indices, :]

        # Get final representation of self and interaction features
        interaction_features = node_feature_store[:, -node_feature_dim:]
        combined_features = self.fc_combine_interaction(torch.cat([node_features_enc, interaction_features], dim=1))

        # Get edge embeddings from node embeddings
        source_node_enc = combined_features[from_idx]
        dest_node_enc  = combined_features[to_idx]
        forward_edge_input = torch.cat((source_node_enc, dest_node_enc, edge_features_enc), dim=-1)
        backward_edge_input = torch.cat((dest_node_enc, source_node_enc, edge_features_enc), dim=-1)
        forward_edge_msg = self.prop_layer._message_net(forward_edge_input)
        backward_edge_msg = self.prop_layer._reverse_message_net(backward_edge_input)
        edge_msg_enc = forward_edge_msg + backward_edge_msg
        edge_counts  = self.fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))

        edge_feature_enc_split = torch.split(edge_msg_enc, edge_counts, dim=0)
        edge_feature_enc_query = edge_feature_enc_split[0::2]
        edge_feature_enc_corpus = edge_feature_enc_split[1::2]

        stacked_qedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size_edge-x.shape[0])) \
                                         for x in edge_feature_enc_query])
        stacked_cedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size_edge-x.shape[0])) \
                                         for x in edge_feature_enc_corpus])
        edge_count_pairs = list(zip(edge_counts[0::2],edge_counts[1::2]))
        from_idx_split = torch.split(from_idx, edge_counts, dim=0)
        to_idx_split = torch.split(to_idx, edge_counts, dim=0)
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
        transport_plan_edge = pytorch_sinkhorn_iters(self.av,transport_plan_edge)

        scores_node_align = -torch.sum(torch.maximum(
            stacked_qnode_final_emb - transport_plan_node@stacked_cnode_final_emb,
            cudavar(self.av,torch.tensor([0]))),
           dim=(1,2))

        scores_kronecker_edge_align = cudavar(self.av,torch.tensor(self.av.consistency_lambda)) * (-torch.sum(torch.maximum(stacked_qedge_emb - transport_plan_edge@stacked_cedge_emb,\
cudavar(self.av,torch.tensor([0]))),dim=(1,2)))
        
        return scores_node_align + scores_kronecker_edge_align