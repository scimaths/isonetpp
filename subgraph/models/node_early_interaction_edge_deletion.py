import torch
import torch.nn.functional as F
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen

class NodeEarlyInteractionEdgeDeletion(torch.nn.Module):
    def __init__(self, av, config, input_dim):
        super(NodeEarlyInteractionEdgeDeletion, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.device = 'cuda' if av.has_cuda and av.want_cuda else 'cpu'
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
        self.lambd = self.config["node_early_interaction_edge_deletion"]["lambd"]
    
    def build_masking_utility(self):
        self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))).to(self.device) for x in range(0,self.max_set_size+1)]

    def get_graph(self, batch):
        graph = batch
        node_features = torch.from_numpy(graph.node_features).to(self.device)
        edge_features = torch.from_numpy(graph.edge_features).to(self.device)
        from_idx = torch.from_numpy(graph.from_idx).long().to(self.device)
        to_idx = torch.from_numpy(graph.to_idx).long().to(self.device)
        graph_idx = torch.from_numpy(graph.graph_idx).long().to(self.device)
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

        mask_from_idx = None

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
                    node_features_enc = self.prop_layer(combined_features, from_idx, to_idx, edge_features_enc, mask_from_idx = mask_from_idx)
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
            transport_plan = pytorch_sinkhorn_iters(self.av, sinkhorn_input)

            # Calculate interpretability
            temp_mask_from_idx = torch.sum(transport_plan * qgraph_mask[:, :, :self.max_set_size], dim=1)
            temp_mask_from_idx = torch.stack((
                torch.ones(temp_mask_from_idx.shape, device=self.device), 
                self.lambd * torch.ones(temp_mask_from_idx.shape).to(temp_mask_from_idx.device) + (1 - self.lambd) * temp_mask_from_idx
            ), dim = 1).view(temp_mask_from_idx.shape[0] * 2, temp_mask_from_idx.shape[1]).flatten()
            mask_from_idx = torch.cat(torch.split(
                temp_mask_from_idx,
                torch.stack((torch.tensor(batch_data_sizes_flat), self.max_set_size - torch.tensor(batch_data_sizes_flat)),
                            dim = 1).view(2 * len(batch_data_sizes_flat)).tolist(),
                dim=0
            )[0::2])

            # Compute interaction
            qnode_features_from_cnodes = torch.bmm(transport_plan, stacked_cnode_store_emb)
            cnode_features_from_qnodes = torch.bmm(transport_plan.permute(0, 2, 1), stacked_qnode_store_emb)
            interleaved_node_features = torch.cat([
                qnode_features_from_cnodes.unsqueeze(1),
                cnode_features_from_qnodes.unsqueeze(1)
            ], dim=1)[:, :, :, node_feature_dim:].reshape(-1, n_prop_update_steps * node_feature_dim) 
            node_feature_store[:, node_feature_dim:] = interleaved_node_features[node_indices, :]

        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum(torch.maximum(
            stacked_qnode_final_emb - transport_plan@stacked_cnode_final_emb,
            torch.tensor([0], device=self.device)),
           dim=(1,2))
        
        return scores