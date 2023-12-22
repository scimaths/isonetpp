import torch
import torch.nn.functional as F
from subgraph.utils import cudavar
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen

class NodeEarlyInteractionAdding(torch.nn.Module):
    def __init__(self, av, config, input_dim):
        super(NodeEarlyInteractionAdding, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
    
    def build_masking_utility(self):
        self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_set_size+1)]
        # ----------------------- obtaining max edges possible -------------
        self.max_edge_size = self.av.MAX_EDGES

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

    # ------------------------------ Function to obtain edge counts from isonet ---------------------
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

    def forward(self, batch_data, batch_data_sizes, batch_adj):
        qgraph_sizes, cgraph_sizes = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av, torch.tensor(qgraph_sizes))
        device = qgraph_sizes.device
        cgraph_sizes = torch.tensor(cgraph_sizes, device=device)
        batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
        batch_data_sizes_flat_tensor = torch.tensor(batch_data_sizes_flat, device=device, dtype=torch.long)
        cumulative_sizes = torch.cumsum(torch.tensor(self.max_set_size, dtype=torch.long, device=device).repeat(len(batch_data_sizes_flat_tensor)), dim=0)

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)

        


        #------------------------------------ Adding (q->q_pad) -------------------------------------
        batch_size = len(batch_data_sizes)
        
        # Calculating number of nodes to add; It is (corpus-query)
        num_nodes_query_existant = [sublist[0] for sublist in batch_data_sizes]
        num_new_nodes_batch = [ (batch_data_sizes[idx][1] - batch_data_sizes[idx][0]) for idx in range(batch_size)]
        
        # Adding features for extra nodes; All the node features are 1 initialized
        num_nodes_initial, _ = node_features.shape
        node_features = torch.ones(num_nodes_initial + sum(num_new_nodes_batch), device=device).reshape(-1, 1)
        
        # Edge counts for each query and corpus graph
        edge_counts_initial  = self.fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
        assert len(edge_counts_initial) == 2 * batch_size
        
        # Calculating number of edges to add; It is (corpus-query)*query (from every query nodes to added nodes)
        num_new_edges = [ num_new_nodes_batch[query_idx] * num_nodes_query_existant[query_idx] for query_idx in range(batch_size)]
        
        # Adding features for extra edges; All the edge features are 1 initialized
        num_edges_initial, _ = edge_features.shape
        edge_features = torch.ones(num_edges_initial + sum(num_new_edges)).reshape(-1, 1).to(device)

        # Increasing index of existant edges; Nodes of graph k are named starting from sum(nodes[:k]); As new nodes are added this has to be changed 
        increase_idx = torch.cat([torch.tensor(sum(num_new_nodes_batch[:((idx + 1) // 2)]), device=device).repeat(edge_counts_initial[idx]) for idx in range(2*batch_size)])
        from_idx += increase_idx
        to_idx += increase_idx

        # Updating batch_data_sizes to have same nodes for query and corpus in a tuple
        batch_data_sizes_flat = [sublist[1] for sublist in batch_data_sizes for _ in sublist]
        batch_data_sizes_flat_tensor = torch.tensor(batch_data_sizes_flat, device=device, dtype=torch.long)
        cumulative_sizes = torch.cumsum(torch.tensor(self.max_set_size, dtype=torch.long, device=device).repeat(len(batch_data_sizes_flat_tensor)), dim=0)

        # Adding new from/to idx
        new_from_idx_query = []
        new_to_idx_query = []

        for query_idx in range(batch_size):
            # Offset to add to each node
            offset = sum(batch_data_sizes_flat[:2*query_idx])
            # From idx are the existant nodes; Pattern: 0 0 0 1 1 1 2 2 2
            new_from_idx = offset + torch.arange(num_nodes_query_existant[query_idx], device=device).unsqueeze(1).repeat(1, num_new_nodes_batch[query_idx]).flatten()
            new_from_idx_query.append(new_from_idx)
            # To idx are the added nodes; Pattern: 0 1 2 0 1 2 0 1 2
            new_to_idx = offset + torch.arange(num_nodes_query_existant[query_idx], num_nodes_query_existant[query_idx] + num_new_nodes_batch[query_idx], device=device).repeat(num_nodes_query_existant[query_idx])
            new_to_idx_query.append(new_to_idx)

        new_from_idx = torch.cat(new_from_idx_query)
        new_to_idx = torch.cat(new_to_idx_query)
        from_idx = torch.cat((from_idx, new_from_idx))
        to_idx = torch.cat((to_idx, new_to_idx))

        # Assigning the graph indices for new nodes
        new_graph_idx = [2*query_idx for query_idx in range(batch_size) for _ in range(num_new_nodes_batch[query_idx])]
        graph_idx = torch.cat((graph_idx, torch.tensor(new_graph_idx, device=device)))
        


        #---------------- Defining the MULT_FACTOR to be be multiplied -------------------
        # Initially, The factor is 1 if the nodes are existant; 0 if nodes are added
        mult_factor = torch.cat((torch.ones(num_edges_initial, device=device), torch.zeros(sum(num_new_edges), device=device)))
        
        # Obtaining the node indices of edges in corpus to index in transport plan (need a offset of max set size)
        corpus_from_idx = torch.cat(torch.split(from_idx[:num_edges_initial], edge_counts_initial)[1::2])
        corpus_to_idx = torch.cat(torch.split(to_idx[:num_edges_initial], edge_counts_initial)[1::2])
        sub_offset = torch.cat([torch.tensor(sum(batch_data_sizes_flat[:2*corpus_idx+1]), device=device).repeat(edge_counts_initial[2*corpus_idx+1]) for corpus_idx in range(batch_size)])
        add_offset = torch.cat([torch.tensor(self.max_set_size*corpus_idx, device=device).repeat(edge_counts_initial[2*corpus_idx+1]) for corpus_idx in range(batch_size)])
        corpus_from_idx += add_offset - sub_offset
        corpus_to_idx += add_offset - sub_offset

        # Obtaining the node indices of edges in added to index in transport plan (need a offset of max set size)
        added_from_idx = from_idx[num_edges_initial:]
        added_to_idx = to_idx[num_edges_initial:]
        sub_offset = torch.cat([torch.tensor(sum(batch_data_sizes_flat[:2*query_idx]), device=device).repeat(num_new_edges[query_idx]) for query_idx in range(batch_size)])
        add_offset = torch.cat([torch.tensor(self.max_set_size*query_idx, device=device).repeat(num_new_edges[query_idx]) for query_idx in range(batch_size)])
        added_from_idx += add_offset - sub_offset
        added_to_idx += add_offset - sub_offset



        
        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        num_nodes, node_feature_dim = encoded_node_features.shape
        batch_size = len(batch_data_sizes)

        n_time_update_steps = self.config['early_interaction']['n_time_updates']
        n_prop_update_steps = self.config['graph_embedding_net']['n_prop_layers']

        node_feature_store = torch.zeros(num_nodes, node_feature_dim * (n_prop_update_steps + 1), device=node_features.device)
        updated_node_feature_store = torch.zeros_like(node_feature_store)

        max_set_size_arange = torch.arange(self.max_set_size, dtype=torch.long, device=device).reshape(1, -1).repeat(batch_size * 2, 1)
        node_presence_mask = max_set_size_arange < batch_data_sizes_flat_tensor.unsqueeze(1)
        max_set_size_arange[1:, ] += cumulative_sizes[:-1].unsqueeze(1)
        node_indices = max_set_size_arange[node_presence_mask]

        for time_idx in range(1, n_time_update_steps + 1):
            node_features_enc = torch.clone(encoded_node_features)
            edge_features_enc = torch.clone(encoded_edge_features)
            for prop_idx in range(1, n_prop_update_steps + 1) :
                nf_idx = node_feature_dim * prop_idx
                interaction_features = node_feature_store[:, nf_idx - node_feature_dim : nf_idx]
                
                combined_features = self.fc_combine_interaction(torch.cat([node_features_enc, interaction_features], dim=1))
                node_features_enc = self.prop_layer(combined_features, from_idx, to_idx, edge_features_enc, mask_from_idx=mult_factor)
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
            
            qgraph_mask = cudavar(self.av, torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_sizes]))
            cgraph_mask = cudavar(self.av, torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_sizes]))
            masked_qnode_final_emb = torch.mul(qgraph_mask,transformed_qnode_final_emb)
            masked_cnode_final_emb = torch.mul(cgraph_mask,transformed_cnode_final_emb)

            sinkhorn_input = torch.matmul(masked_qnode_final_emb, masked_cnode_final_emb.permute(0, 2, 1))
            transport_plan = pytorch_sinkhorn_iters(self.av, sinkhorn_input)

            # Compute interaction
            qnode_features_from_cnodes = torch.bmm(transport_plan, stacked_cnode_store_emb)
            cnode_features_from_qnodes = torch.bmm(transport_plan.permute(0, 2, 1), stacked_qnode_store_emb)
            interleaved_node_features = torch.cat([
                qnode_features_from_cnodes.unsqueeze(1),
                cnode_features_from_qnodes.unsqueeze(1)
            ], dim=1)[:, :, :, node_feature_dim:].reshape(-1, n_prop_update_steps * node_feature_dim) 
            node_feature_store[:, node_feature_dim:] = interleaved_node_features[node_indices, :]



            
            # ---------------------------------- Update MULT_FACTOR --------------------------------------------------
            # Layout transport plan (B,Q,C) -> (Q,B*C)
            layout_transport_plan = transport_plan.transpose(0, 1).flatten(1)

            # Obtaining special from/to index with offsets of max_node_size
            special_from = layout_transport_plan[:, corpus_from_idx]
            special_to = layout_transport_plan[:, corpus_to_idx]

            special_from = torch.split(special_from, edge_counts_initial[1::2], dim=1)
            special_from = torch.stack([F.pad(x, pad=(0,self.max_edge_size-x.shape[1])) \
                                            for x in special_from])
            special_from = special_from.reshape(-1, self.max_edge_size)
            
            special_to = torch.split(special_to, edge_counts_initial[1::2], dim=1)
            special_to = torch.stack([F.pad(x, pad=(0,self.max_edge_size-x.shape[1])) \
                                            for x in special_to])
            special_to = special_to.reshape(-1, self.max_edge_size)
            
            intermediate_special_from = special_from[added_from_idx, :]
            intermediate_special_to = special_to[added_to_idx, :]
            mult_factor[-sum(num_new_edges):] = torch.sum(intermediate_special_from * intermediate_special_to, dim=1)

            intermediate_special_from = special_from[added_to_idx, :]
            intermediate_special_to = special_to[added_from_idx, :]
            mult_factor[-sum(num_new_edges):] += torch.sum(intermediate_special_from * intermediate_special_to, dim=1)
        



        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum(torch.maximum(
            stacked_qnode_final_emb - transport_plan@stacked_cnode_final_emb,
            cudavar(self.av,torch.tensor([0]))),
           dim=(1,2))
        
        return scores