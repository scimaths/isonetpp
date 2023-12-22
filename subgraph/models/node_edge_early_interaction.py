import torch
import torch.nn.functional as F
from subgraph.utils import cudavar
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen

def batched_kron(a, b):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)

class NodeEdgeEarlyInteraction(torch.nn.Module):
    def __init__(self, av, config, input_dim):
        super(NodeEdgeEarlyInteraction, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.device = 'cuda' if av.has_cuda and av.want_cuda else 'cpu'
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
    
    def build_masking_utility(self):
        self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.max_set_size_edge = self.av.MAX_EDGES
        self.graph_size_to_mask_map = [torch.cat((
            torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim),
            torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim)
        )).to(self.device) for x in range(0, self.max_set_size+1)]

    def get_graph(self, batch):
        graph = batch
        node_features = torch.from_numpy(graph.node_features).to(self.device)
        edge_features = torch.from_numpy(graph.edge_features).to(self.device)
        from_idx = torch.from_numpy(graph.from_idx).long().to(self.device)
        to_idx = torch.from_numpy(graph.to_idx).long().to(self.device)
        graph_idx = torch.from_numpy(graph.graph_idx).long().to(self.device)
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def fetch_edge_counts(self,to_idx,from_idx,graph_idx,num_graphs):
        from GMN.segment import unsorted_segment_sum
        to_edges = unsorted_segment_sum(cudavar(self.av,torch.ones(len(to_idx))), to_idx, len(graph_idx))
        from_edges = unsorted_segment_sum(cudavar(self.av,torch.ones(len(from_idx))), from_idx, len(graph_idx))
        to_edge_count = unsorted_segment_sum(to_edges, graph_idx, num_graphs)
        from_edge_count = unsorted_segment_sum(from_edges, graph_idx, num_graphs)
        assert(to_edge_count == from_edge_count).all()
        assert(sum(to_edge_count) == len(to_idx))
        return list(map(int, to_edge_count.tolist()))

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
        qgraph_sizes = torch.tensor(qgraph_sizes, device=self.device)
        cgraph_sizes = torch.tensor(cgraph_sizes, device=self.device)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]
        batch_data_sizes_flat_tensor = torch.tensor(batch_data_sizes_flat, device=self.device, dtype=torch.long)
        cumulative_sizes = torch.cumsum(torch.tensor(self.max_set_size, dtype=torch.long, device=self.device).repeat(len(batch_data_sizes_flat_tensor)), dim=0)

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
        
        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        num_nodes, node_feature_dim = encoded_node_features.shape
        batch_size = len(batch_data_sizes)

        n_time_update_steps = self.config['early_interaction']['n_time_updates']
        n_prop_update_steps = self.config['graph_embedding_net']['n_prop_layers']

        node_feature_store = torch.zeros(num_nodes, node_feature_dim * (n_prop_update_steps + 1), device=node_features.device)
        updated_node_feature_store = torch.zeros_like(node_feature_store)

        max_set_size_arange = torch.arange(self.max_set_size, dtype=torch.long, device=self.device).reshape(1, -1).repeat(batch_size * 2, 1)
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
                # if self.config['early_interaction']['prop_separate_params']:
                #     node_features_enc = self.prop_layers[time_idx - 1](combined_features, from_idx, to_idx, edge_features_enc)
                # else:
                node_features_enc, messages = self.prop_layer(combined_features, from_idx, to_idx, edge_features_enc, return_msg=True)
                updated_node_feature_store[:, nf_idx : nf_idx + node_feature_dim] = torch.clone(node_features_enc)

            node_feature_store = torch.clone(updated_node_feature_store)

            node_feature_store_split = torch.split(node_feature_store, batch_data_sizes_flat, dim=0)
            node_feature_store_query = node_feature_store_split[0::2]
            node_feature_store_corpus = node_feature_store_split[1::2]

            stacked_qnode_store_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) for x in node_feature_store_query])
            stacked_cnode_store_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) for x in node_feature_store_corpus])
            
            # Compute transport plan
            stacked_qnode_final_emb = stacked_qnode_store_emb[:,:,-node_feature_dim:]
            stacked_cnode_final_emb = stacked_cnode_store_emb[:,:,-node_feature_dim:]
            transformed_qnode_final_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qnode_final_emb)))
            transformed_cnode_final_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cnode_final_emb)))
            
            qgraph_mask = torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_sizes])
            cgraph_mask = torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_sizes])
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

        if self.diagnostic_mode:
            return node_transport_plan

        # Get edge embeddings from node embeddings
        edge_counts  = self.fetch_edge_counts(to_idx, from_idx, graph_idx, 2*len(batch_data_sizes))
        edge_feature_enc_split = torch.split(messages, edge_counts, dim=0)
        edge_feature_enc_query = edge_feature_enc_split[0::2]
        edge_feature_enc_corpus = edge_feature_enc_split[1::2]

        # Get transport plan on edges
        stacked_qedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size_edge-x.shape[0])) \
                                         for x in edge_feature_enc_query])
        stacked_cedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size_edge-x.shape[0])) \
                                         for x in edge_feature_enc_corpus])
        # Consistency scoring of node and edge alignment
        edge_count_pairs = list(zip(edge_counts[0::2], edge_counts[1::2]))
        from_idx_split = torch.split(from_idx, edge_counts, dim=0)
        to_idx_split = torch.split(to_idx, edge_counts, dim=0)
        prefix_sum_node_counts = [sum(batch_data_sizes_flat[:k]) for k in range(len(batch_data_sizes_flat))]
        to_idx_split_relabeled = [x1 - x2 for (x1, x2) in zip(to_idx_split, prefix_sum_node_counts)]
        from_idx_split_relabeled = [x1 - x2 for (x1, x2) in zip(from_idx_split, prefix_sum_node_counts)]

        #=================STRAIGHT=============
        from_node_ids_mapping = [torch.cartesian_prod(x,y) \
           for (x,y) in zip(from_idx_split_relabeled[0::2], from_idx_split_relabeled[1::2])]
        from_node_map_scores = [x[y[:,0],y[:,1]].view(z) for (x,y,z) in zip(node_transport_plan,from_node_ids_mapping,edge_count_pairs)]
        to_node_ids_mapping = [torch.cartesian_prod(x,y) \
           for (x,y) in zip(to_idx_split_relabeled[0::2], to_idx_split_relabeled[1::2])]
        to_node_map_scores = [x[y[:,0],y[:,1]].view(z) for (x,y,z) in zip(node_transport_plan,to_node_ids_mapping,edge_count_pairs)]
        stacked_from_node_map_scores = torch.stack([F.pad(x, pad=(0,self.max_set_size_edge-x.shape[1],0,self.max_set_size_edge-x.shape[0])) \
                                 for x in from_node_map_scores])
        stacked_to_node_map_scores = torch.stack([F.pad(x, pad=(0,self.max_set_size_edge-x.shape[1],0,self.max_set_size_edge-x.shape[0])) \
                                 for x in to_node_map_scores])
        stacked_all_node_map_scores = torch.mul(stacked_from_node_map_scores,stacked_to_node_map_scores)

        #==================CROSS=========
        from_node_ids_mapping1 = [torch.cartesian_prod(x,y) \
           for (x,y) in zip(from_idx_split_relabeled[0::2], to_idx_split_relabeled[1::2])]
        from_node_map_scores1 = [x[y[:,0],y[:,1]].view(z) for (x,y,z) in zip(node_transport_plan,from_node_ids_mapping1,edge_count_pairs)]
        to_node_ids_mapping1 = [torch.cartesian_prod(x,y) \
           for (x,y) in zip(to_idx_split_relabeled[0::2], from_idx_split_relabeled[1::2])]
        to_node_map_scores1 = [x[y[:,0],y[:,1]].view(z) for (x,y,z) in zip(node_transport_plan,to_node_ids_mapping1,edge_count_pairs)]
        stacked_from_node_map_scores_cross = torch.stack([F.pad(x, pad=(0,self.max_set_size_edge-x.shape[1],0,self.max_set_size_edge-x.shape[0])) \
                                 for x in from_node_map_scores1])
        stacked_to_node_map_scores_cross = torch.stack([F.pad(x, pad=(0,self.max_set_size_edge-x.shape[1],0,self.max_set_size_edge-x.shape[0])) \
                                 for x in to_node_map_scores1])
        stacked_all_node_map_scores_cross = torch.mul(stacked_from_node_map_scores_cross,stacked_to_node_map_scores_cross)

        final_stacked_all_node_map_scores = torch.cat((
            torch.cat((stacked_all_node_map_scores, stacked_all_node_map_scores_cross),dim=-1),
            torch.cat((stacked_all_node_map_scores_cross, stacked_all_node_map_scores),dim=-1)
        ), dim=-2)

        doubled_stacked_qedge_emb = stacked_qedge_emb.repeat(1,2,1)
        doubled_stacked_cedge_emb = stacked_cedge_emb.repeat(1,2,1)

        scores_node_align = -torch.sum(torch.maximum(
            stacked_qnode_final_emb - node_transport_plan@stacked_cnode_final_emb,
            cudavar(self.av,torch.tensor([0]))),
           dim=(1,2))

        regularizer_consistency = -torch.sum((
            doubled_stacked_qedge_emb - final_stacked_all_node_map_scores@doubled_stacked_cedge_emb),
        dim=(1,2))

        return scores_node_align + self.av.consistency_lambda * regularizer_consistency