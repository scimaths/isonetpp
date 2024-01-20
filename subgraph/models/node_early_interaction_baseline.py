import torch
import torch.nn.functional as F
from subgraph.utils import cudavar
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen

class NodeEarlyInteractionBaseline(torch.nn.Module):
    def __init__(self, av, config, input_dim):
        super(NodeEarlyInteractionBaseline, self).__init__()
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
        cumulative_sizes = torch.cumsum(torch.tensor(self.max_set_size, dtype=torch.long, device=device).repeat(len(batch_data_sizes_flat_tensor)), dim=0)

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
        
        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        num_nodes, node_feature_dim = encoded_node_features.shape
        batch_size = len(batch_data_sizes)

        n_prop_update_steps = self.config['graph_embedding_net']['n_prop_layers']

        node_feature_store = torch.zeros(num_nodes, node_feature_dim, device=node_features.device)

        max_set_size_arange = torch.arange(self.max_set_size, dtype=torch.long, device=device).reshape(1, -1).repeat(batch_size * 2, 1)
        node_presence_mask = max_set_size_arange < batch_data_sizes_flat_tensor.unsqueeze(1)
        max_set_size_arange[1:, ] += cumulative_sizes[:-1].unsqueeze(1)
        node_indices = max_set_size_arange[node_presence_mask]

        node_features_enc = torch.clone(encoded_node_features)
        edge_features_enc = torch.clone(encoded_edge_features)
        for prop_idx in range(1, n_prop_update_steps + 1) :
            
            combined_features = self.fc_combine_interaction(torch.cat([node_features_enc, node_feature_store], dim=1))
            
            node_features_enc = self.prop_layer(combined_features, from_idx, to_idx, edge_features_enc)

            node_feature_store_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
            node_feature_store_query = node_feature_store_split[0::2]
            node_feature_store_corpus = node_feature_store_split[1::2]

            stacked_qnode_store_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) for x in node_feature_store_query])
            stacked_cnode_store_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) for x in node_feature_store_corpus])
            
            # Compute transport plan
            transformed_qnode_final_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qnode_store_emb)))
            transformed_cnode_final_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cnode_store_emb)))
            
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
            ], dim=1).reshape(-1, node_feature_dim) 
            node_feature_store = interleaved_node_features[node_indices, :]

        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum(torch.maximum(
            stacked_qnode_store_emb - transport_plan@stacked_cnode_store_emb,
            cudavar(self.av,torch.tensor([0]))),
           dim=(1,2))
        
        return scores