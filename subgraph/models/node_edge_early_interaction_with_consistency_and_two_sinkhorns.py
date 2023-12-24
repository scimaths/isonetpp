import torch
import torch.nn.functional as F
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen

class NodeEdgeEarlyInteractionWithConsistencyAndTwoSinkhorns(torch.nn.Module):
    def __init__(self, av, config, input_dim):
        super(NodeEdgeEarlyInteractionWithConsistencyAndTwoSinkhorns, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.device = 'cuda' if av.has_cuda and av.want_cuda else 'cpu'
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
    
    def build_masking_utility(self):
        ###### MASKS FOR NODES ######
        self.max_node_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        #this mask pattern sets bottom last few rows to 0 based on padding needs
        self.graph_size_to_mask_map_node = [torch.cat((
            torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim),
            torch.tensor([0]).repeat(self.max_node_set_size-x,1).repeat(1,self.av.transform_dim)
        )).to(self.device) for x in range(0,self.max_node_set_size+1)]
        # Mask pattern sets top left (k)*(k) square to 1 inside arrays of size n*n. Rest elements are 0
        self.set_size_to_mask_map_node = [torch.cat((
            torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([x,self.max_node_set_size-x])).repeat(x,1),
            torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([0,self.max_node_set_size])).repeat(self.max_node_set_size-x,1)
        )) for x in range(0,self.max_node_set_size+1)]

        ###### MASKS FOR EDGES ######
        self.max_edge_set_size = self.av.MAX_EDGES
        #this mask pattern sets bottom last few rows to 0 based on padding needs
        self.graph_size_to_mask_map_edge = [torch.cat((
            torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), 
            torch.tensor([0]).repeat(self.max_edge_set_size-x,1).repeat(1,self.av.transform_dim)
        )).to(self.device) for x in range(0,self.max_edge_set_size+1)]
        # Mask pattern sets top left (k)*(k) square to 1 inside arrays of size n*n. Rest elements are 0
        self.set_size_to_mask_map_edge = [torch.cat((
            torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([x,self.max_edge_set_size-x])).repeat(x,1),
            torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([0,self.max_edge_set_size])).repeat(self.max_edge_set_size-x,1)
        )) for x in range(0,self.max_edge_set_size+1)]


    def fetch_edge_counts(self,to_idx,from_idx,graph_idx,num_graphs):
        #HACK - since I'm not storing edge sizes of each graph (only storing node sizes)
        #and no. of nodes is not equal to no. of edges
        #so a hack to obtain no of edges in each graph from available info
        from GMN.segment import unsorted_segment_sum
        tt = unsorted_segment_sum(torch.ones(len(to_idx), device=self.device), to_idx, len(graph_idx))
        tt1 = unsorted_segment_sum(torch.ones(len(from_idx), device=self.device), from_idx, len(graph_idx))
        edge_counts = unsorted_segment_sum(tt, graph_idx, num_graphs)
        edge_counts1 = unsorted_segment_sum(tt1, graph_idx, num_graphs)
        assert(edge_counts == edge_counts1).all()
        assert(sum(edge_counts)== len(to_idx))
        return list(map(int,edge_counts.tolist()))

    def get_graph(self, batch):
        graph = batch
        node_features = torch.from_numpy(graph.node_features).to(self.device)
        edge_features = torch.from_numpy(graph.edge_features).to(self.device)
        from_idx = torch.from_numpy(graph.from_idx).long().to(self.device)
        to_idx = torch.from_numpy(graph.to_idx).long().to(self.device)
        graph_idx = torch.from_numpy(graph.graph_idx).long().to(self.device)
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def build_layers(self):
        self.consistency_lambda = self.av.consistency_lambda

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        self.prop_config = self.config['graph_embedding_net'].copy()
        self.prop_config.pop('n_prop_layers',None)
        self.prop_config.pop('share_prop_params',None)
        self.final_edge_encoding_dim = 30
        self.prop_config['final_edge_encoding_dim'] = self.final_edge_encoding_dim
        self.message_feature_dim = self.prop_config['edge_hidden_sizes'][-1]
        self.prop_layer = gmngen.GraphPropLayer(**self.prop_config)
        
        edge_combined_feature_dim = self.message_feature_dim + self.config['encoder']['edge_feature_dim']
        self.fc_combine_interaction_edge = torch.nn.Sequential(
            torch.nn.Linear(edge_combined_feature_dim, edge_combined_feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_combined_feature_dim, self.final_edge_encoding_dim)
        )

        node_combined_feature_dim = 2 * self.prop_config['node_state_dim']
        self.fc_combine_interaction_node = torch.nn.Sequential(
            torch.nn.Linear(node_combined_feature_dim, node_combined_feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_combined_feature_dim, self.prop_config['node_state_dim'])
        )

        self.fc_transform_node = torch.nn.Sequential(
            torch.nn.Linear(self.av.filters_3, self.av.transform_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        )

        double_encoding_size = self.message_feature_dim * 2
        self.fc_combine_double_edge_encoding = torch.nn.Sequential(
            torch.nn.Linear(double_encoding_size, double_encoding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(double_encoding_size, self.message_feature_dim)
        )

    def forward(self, batch_data, batch_data_sizes, batch_adj):
        qgraph_sizes, cgraph_sizes = zip(*batch_data_sizes)
        qgraph_sizes = torch.tensor(qgraph_sizes, device=self.device)
        cgraph_sizes = torch.tensor(cgraph_sizes, device=self.device)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]
        batch_data_sizes_flat_tensor = torch.tensor(batch_data_sizes_flat, device=self.device, dtype=torch.long)
        cumulative_node_sizes = torch.cumsum(torch.tensor(self.max_node_set_size, dtype=torch.long, device=self.device).repeat(len(batch_data_sizes_flat_tensor)), dim=0)
        cumulative_edge_sizes = torch.cumsum(torch.tensor(self.max_edge_set_size, dtype=torch.long, device=self.device).repeat(len(batch_data_sizes_flat_tensor)), dim=0)

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)

        # Get counts of nodes and edges
        edge_counts = self.fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
        edge_counts_tensor = torch.tensor(edge_counts, device=self.device, dtype=torch.long)
        
        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        num_nodes, node_feature_dim = encoded_node_features.shape
        num_edges, _ = encoded_edge_features.shape
        batch_size = len(batch_data_sizes)

        n_time_update_steps = self.config['early_interaction']['n_time_updates']
        n_prop_update_steps = self.config['graph_embedding_net']['n_prop_layers']

        # Utility data for edge transport plan computation
        edge_count_pairs = list(zip(edge_counts[0::2], edge_counts[1::2]))
        from_idx_split = torch.split(from_idx, edge_counts, dim=0)
        to_idx_split = torch.split(to_idx, edge_counts, dim=0)
        prefix_sum_node_counts = [sum(batch_data_sizes_flat[:k]) for k in range(len(batch_data_sizes_flat))]
        to_idx_split_relabeled = [x1 - x2 for (x1, x2) in zip(to_idx_split, prefix_sum_node_counts)]
        from_idx_split_relabeled = [x1 - x2 for (x1, x2) in zip(from_idx_split, prefix_sum_node_counts)]

        from_node_ids_mapping_straight = [torch.cartesian_prod(x,y) for (x,y) in zip(from_idx_split_relabeled[0::2], from_idx_split_relabeled[1::2])]
        to_node_ids_mapping_straight = [torch.cartesian_prod(x,y) for (x,y) in zip(to_idx_split_relabeled[0::2], to_idx_split_relabeled[1::2])]

        from_node_ids_mapping_cross = [torch.cartesian_prod(x,y) for (x,y) in zip(from_idx_split_relabeled[0::2], to_idx_split_relabeled[1::2])]
        to_node_ids_mapping_cross = [torch.cartesian_prod(x,y) for (x,y) in zip(to_idx_split_relabeled[0::2], from_idx_split_relabeled[1::2])]

        # Create feature stores
        edge_feature_store = torch.zeros(num_edges, self.message_feature_dim * (n_prop_update_steps + 1), device=node_features.device)
        updated_edge_feature_store = torch.zeros_like(edge_feature_store)
        node_feature_store = torch.zeros(num_nodes, node_feature_dim * (n_prop_update_steps + 1), device=node_features.device)
        updated_node_feature_store = torch.zeros_like(node_feature_store)

        # Get presence mask and indices for nodes
        max_node_set_size_arange = torch.arange(self.max_node_set_size, dtype=torch.long, device=self.device).reshape(1, -1).repeat(batch_size * 2, 1)
        node_presence_mask = max_node_set_size_arange < batch_data_sizes_flat_tensor.unsqueeze(1)
        max_node_set_size_arange[1:, ] += cumulative_node_sizes[:-1].unsqueeze(1)
        node_indices = max_node_set_size_arange[node_presence_mask]

        # Get presence mask and indices for edges
        max_edge_set_size_arange = torch.arange(self.max_edge_set_size, dtype=torch.long, device=self.device).reshape(1, -1).repeat(batch_size * 2, 1)
        edge_presence_mask = max_edge_set_size_arange < edge_counts_tensor.unsqueeze(1)
        max_edge_set_size_arange[1:, ] += cumulative_edge_sizes[:-1].unsqueeze(1)
        edge_indices = max_edge_set_size_arange[edge_presence_mask]

        for time_idx in range(1, n_time_update_steps + 1):
            node_features_enc = torch.clone(encoded_node_features)
            edge_features_enc = torch.clone(encoded_edge_features)

            # Compute \bar{h}[0, 1] in node_combined_features
            node_interaction_features = node_feature_store[:, :node_feature_dim] # h(opp)[0, 0]; opp refers to c for q and vice-versa
            node_combined_features = self.fc_combine_interaction_node(torch.cat([node_features_enc, node_interaction_features], dim=1))

            for prop_idx in range(1, n_prop_update_steps + 1) :
                nf_idx_edge = self.message_feature_dim * prop_idx
                nf_idx_node = node_feature_dim * prop_idx

                # Compute mlp = MLP_\phi( \sum r(opp)[k, t] ) in edge_combined_features
                edge_interaction_features = edge_feature_store[:, nf_idx_edge - self.message_feature_dim : nf_idx_edge]
                edge_combined_features = self.fc_combine_interaction_edge(torch.cat([edge_features_enc, edge_interaction_features], dim=1))
                
                # Compute h[k+1,t+1] in node_features_enc using MSG_\phi, AGGR_\phi, COMB_\phi and inputs are \bar{h}[k, t+1], mlp
                node_features_enc = self.prop_layer(node_combined_features, from_idx, to_idx, edge_combined_features)

                # Compute \bar{h}[k+1, t+1] in node_combined_features - REPEATED
                node_interaction_features = node_feature_store[:, nf_idx_node : nf_idx_node + node_feature_dim] # h(opp)[k+1, t]; opp refers to c for q and vice-versa
                node_combined_features = self.fc_combine_interaction_node(torch.cat([node_features_enc, node_interaction_features], dim=1))

                # Compute r[k+1, t+1] in messages
                source_node_enc = node_combined_features[from_idx]
                dest_node_enc  = node_combined_features[to_idx]
                forward_edge_input = torch.cat((source_node_enc, dest_node_enc, edge_combined_features),dim=-1)
                backward_edge_input = torch.cat((dest_node_enc, source_node_enc, edge_combined_features),dim=-1)
                forward_edge_msg = self.prop_layer._message_net(forward_edge_input)
                backward_edge_msg = self.prop_layer._reverse_message_net(backward_edge_input)
                messages = forward_edge_msg + backward_edge_msg
                
                updated_node_feature_store[:, nf_idx_node : nf_idx_node + node_feature_dim] = torch.clone(node_features_enc)
                updated_edge_feature_store[:, nf_idx_edge : nf_idx_edge + self.message_feature_dim] = torch.clone(messages)

            ################################ COMPUTE NODE INTERACTION ################################
            node_feature_store = torch.clone(updated_node_feature_store)

            node_feature_store_split = torch.split(node_feature_store, batch_data_sizes_flat, dim=0)
            node_feature_store_query = node_feature_store_split[0::2]
            node_feature_store_corpus = node_feature_store_split[1::2]

            stacked_qnode_store_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_set_size-x.shape[0])) for x in node_feature_store_query])
            stacked_cnode_store_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_node_set_size-x.shape[0])) for x in node_feature_store_corpus])
            
            # Compute node transport plan
            stacked_qnode_final_emb = stacked_qnode_store_emb[:,:,-node_feature_dim:]
            stacked_cnode_final_emb = stacked_cnode_store_emb[:,:,-node_feature_dim:]
            transformed_qnode_final_emb = self.fc_transform_node(stacked_qnode_final_emb)
            transformed_cnode_final_emb = self.fc_transform_node(stacked_cnode_final_emb)
            
            qgraph_mask = torch.stack([self.graph_size_to_mask_map_node[i] for i in qgraph_sizes])
            cgraph_mask = torch.stack([self.graph_size_to_mask_map_node[i] for i in cgraph_sizes])
            masked_qnode_final_emb = torch.mul(qgraph_mask,transformed_qnode_final_emb)
            masked_cnode_final_emb = torch.mul(cgraph_mask,transformed_cnode_final_emb)

            sinkhorn_input = torch.matmul(masked_qnode_final_emb, masked_cnode_final_emb.permute(0, 2, 1))
            node_transport_plan = pytorch_sinkhorn_iters(self.av, sinkhorn_input)

            # Compute interaction
            qnode_features_from_cnodes = torch.bmm(node_transport_plan, stacked_cnode_store_emb)
            cnode_features_from_qnodes = torch.bmm(node_transport_plan.permute(0, 2, 1), stacked_qnode_store_emb)
            interleaved_node_features = torch.cat([
                qnode_features_from_cnodes.unsqueeze(1),
                cnode_features_from_qnodes.unsqueeze(1)
            ], dim=1)[:, :, :, node_feature_dim:].reshape(-1, n_prop_update_steps * node_feature_dim) 
            node_feature_store[:, node_feature_dim:] = interleaved_node_features[node_indices, :]

            ################################ COMPUTE EDGE INTERACTION ################################
            edge_feature_store = torch.clone(updated_edge_feature_store)

            edge_feature_store_split = torch.split(edge_feature_store, edge_counts, dim=0)
            edge_feature_store_query = edge_feature_store_split[0::2]
            edge_feature_store_corpus = edge_feature_store_split[1::2]

            stacked_qedge_store_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_edge_set_size-x.shape[0])) for x in edge_feature_store_query])
            stacked_cedge_store_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_edge_set_size-x.shape[0])) for x in edge_feature_store_corpus])

            # Compute edge transport plan
            #=================STRAIGHT=============
            from_node_map_scores = [x[y[:,0],y[:,1]].view(z) for (x,y,z) in zip(node_transport_plan, from_node_ids_mapping_straight, edge_count_pairs)]
            to_node_map_scores = [x[y[:,0],y[:,1]].view(z) for (x,y,z) in zip(node_transport_plan, to_node_ids_mapping_straight, edge_count_pairs)]
            stacked_from_node_map_scores = torch.stack([F.pad(x, pad=(0,self.max_edge_set_size-x.shape[1],0,self.max_edge_set_size-x.shape[0])) \
                                    for x in from_node_map_scores])
            stacked_to_node_map_scores = torch.stack([F.pad(x, pad=(0,self.max_edge_set_size-x.shape[1],0,self.max_edge_set_size-x.shape[0])) \
                                    for x in to_node_map_scores])
            stacked_all_node_map_scores_straight = torch.mul(stacked_from_node_map_scores, stacked_to_node_map_scores)

            #==================CROSS=========
            from_node_map_scores_cross = [x[y[:,0],y[:,1]].view(z) for (x,y,z) in zip(node_transport_plan, from_node_ids_mapping_cross, edge_count_pairs)]
            to_node_map_scores_cross = [x[y[:,0],y[:,1]].view(z) for (x,y,z) in zip(node_transport_plan, to_node_ids_mapping_cross, edge_count_pairs)]
            stacked_from_node_map_scores_cross = torch.stack([F.pad(x, pad=(0,self.max_edge_set_size-x.shape[1],0,self.max_edge_set_size-x.shape[0])) \
                                    for x in from_node_map_scores_cross])
            stacked_to_node_map_scores_cross = torch.stack([F.pad(x, pad=(0,self.max_edge_set_size-x.shape[1],0,self.max_edge_set_size-x.shape[0])) \
                                    for x in to_node_map_scores_cross])
            stacked_all_node_map_scores_cross = torch.mul(stacked_from_node_map_scores_cross, stacked_to_node_map_scores_cross)

            edge_transport_plan = pytorch_sinkhorn_iters(
                self.av,
                stacked_all_node_map_scores_straight + stacked_all_node_map_scores_cross
            )

            stacked_qedge_emb_final = stacked_qedge_store_emb[:,:,-self.message_feature_dim:]
            stacked_cedge_emb_final = stacked_cedge_store_emb[:,:,-self.message_feature_dim:]

            qedge_features_from_cedges = torch.bmm(edge_transport_plan, stacked_cedge_store_emb)
            cedge_features_from_qedges = torch.bmm(edge_transport_plan.permute(0, 2, 1), stacked_qedge_store_emb)
            interleaved_edge_features = torch.cat([
                qedge_features_from_cedges.unsqueeze(1),
                cedge_features_from_qedges.unsqueeze(1)
            ], dim=1)[:, :, :, self.message_feature_dim:].reshape(-1, n_prop_update_steps * self.message_feature_dim)
            edge_feature_store[:, self.message_feature_dim:] = interleaved_edge_features[edge_indices, :]

        if self.diagnostic_mode:
            return node_transport_plan

        ################################ COMPUTE SCORE - NODE ONLY ################################
        node_alignment_scores = -torch.sum(torch.maximum(
            stacked_qnode_final_emb - node_transport_plan@stacked_cnode_final_emb,
            torch.tensor([0], device=self.device)
        ), dim=(1,2))

        consistency_regularizer = -torch.sum(torch.maximum(
            stacked_qedge_emb_final - edge_transport_plan@stacked_cedge_emb_final,
            torch.tensor([0], device=self.device)
        ), dim=(1,2))
        
        return node_alignment_scores + self.consistency_lambda * consistency_regularizer
