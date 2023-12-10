import torch
import torch.nn.functional as F
import GMN.graphembeddingnetwork as gmngen
from GMN.segment import unsorted_segment_sum
from subgraph.models.utils import pytorch_sinkhorn_iters

class EdgeEarlyInteraction(torch.nn.Module):
    def __init__(self, av, config, input_dim):
        super(EdgeEarlyInteraction, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.diagnostic_mode = False
        self.device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
        
        self.n_time_update_steps = self.config['early_interaction']['n_time_updates']
        self.n_prop_update_steps = self.config['graph_embedding_net']['n_prop_layers']

        self.build_masking_utility()
        self.build_layers()
    
    def build_masking_utility(self):
        self.max_set_size = self.av.MAX_EDGES

        # Mask bottom rows to 0 based on padding
        self.graph_size_to_mask_map = [torch.cat([
            torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim),
            torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim)]
        ).to(self.device) for x in range(0,self.max_set_size+1)]

        # Mask top left (k)*(k) subsquare to 1 inside arrays of size n*n. Rest elements are 0
        self.set_size_to_mask_map = [torch.cat([
            torch.repeat_interleave(torch.tensor([1,0]), torch.tensor([x,self.max_set_size-x])).repeat(x,1),
            torch.repeat_interleave(torch.tensor([1,0]), torch.tensor([0,self.max_set_size])).repeat(self.max_set_size-x,1)
        ]).to(self.device) for x in range(0,self.max_set_size+1)]

    def fetch_edge_counts(self, to_idx, from_idx, graph_idx, num_graphs):
        # Group edges into src (dest) nodes, then group contributions from nodes 
        edges_per_to_node = unsorted_segment_sum(
            torch.ones(len(to_idx), device=self.device),
            to_idx, len(graph_idx)
        )
        edges_per_from_node = unsorted_segment_sum(
            torch.ones(len(from_idx), device=self.device),
            from_idx, len(graph_idx)
        )
        edge_counts_to = unsorted_segment_sum(edges_per_to_node, graph_idx, num_graphs)
        edge_counts_from = unsorted_segment_sum(edges_per_from_node, graph_idx, num_graphs)
        
        assert(edge_counts_to == edge_counts_from).all()
        assert(sum(edge_counts_to)== len(to_idx))

        return edge_counts_to.long()

    def get_graph(self, graph):
        node_features = torch.tensor(graph.node_features, device=self.device)
        edge_features = torch.tensor(graph.edge_features, device=self.device)
        from_idx = torch.tensor(graph.from_idx, device=self.device).long()
        to_idx = torch.tensor(graph.to_idx, device=self.device).long()
        graph_idx = torch.tensor(graph.graph_idx, device=self.device).long()
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def build_layers(self):
        self.prop_config = self.config['graph_embedding_net'].copy()
        self.prop_config.pop('n_prop_layers',None)
        self.prop_config.pop('share_prop_params',None)
        self.final_edge_encoding_dim = self.config['edge_early_interaction']['hidden_dim']
        self.prop_config['final_edge_encoding_dim'] = self.final_edge_encoding_dim
        self.message_feature_dim = self.prop_config['edge_hidden_sizes'][-1]
        self.prop_layer = gmngen.GraphPropLayer(**self.prop_config)

        encoder_config = self.config['encoder'].copy()
        encoder_config['edge_hidden_sizes'] = [self.message_feature_dim, ]
        self.encoder = gmngen.GraphEncoder(**encoder_config)
        
        self.fc_combine_interaction = torch.nn.Sequential(
            torch.nn.Linear(self.message_feature_dim, self.message_feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.message_feature_dim, self.final_edge_encoding_dim)
        )
        self.fc_transform1 = torch.nn.Linear(2*self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)

    def forward(self, batch_data, batch_data_sizes, batch_adj):
        qgraph_sizes, cgraph_sizes = zip(*batch_data_sizes)
        qgraph_sizes = torch.tensor(qgraph_sizes, device=self.device)
        cgraph_sizes = torch.tensor(cgraph_sizes, device=self.device)
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]
        batch_data_sizes_flat_tensor = torch.tensor(batch_data_sizes_flat, device=self.device, dtype=torch.long)
        cumulative_sizes = self.max_set_size * torch.arange(
            start=1, end=len(batch_data_sizes_flat_tensor) + 1,
            dtype=torch.long, device=self.device
        )

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
        edge_counts  = self.fetch_edge_counts(to_idx, from_idx, graph_idx, 2*len(batch_data_sizes))
        batch_size = len(batch_data_sizes)

        # Create edge-feature-store
        # Saves K * F features per edge, time T is rolled along
        # edge-store (T) ---compute--> updated-edge-store ---save---> edge-store (T+1)
        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        total_edges, _ = encoded_edge_features.shape
        edge_feature_store = torch.zeros(total_edges, self.message_feature_dim * (self.n_prop_update_steps + 1), device=self.device)
        updated_edge_feature_store = torch.zeros_like(edge_feature_store)

        max_set_size_arange = torch.arange(self.max_set_size, dtype=torch.long, device=self.device).reshape(1, -1).repeat(batch_size * 2, 1)
        edge_presence_mask = max_set_size_arange < edge_counts.unsqueeze(1)
        max_set_size_arange[1:, ] += cumulative_sizes[:-1].unsqueeze(1)
        edge_indices = max_set_size_arange[edge_presence_mask]

        for time_idx in range(1, self.n_time_update_steps + 1):
            # Start with base features at every timestep
            node_features_enc = torch.clone(encoded_node_features)
            edge_features_enc = torch.clone(encoded_edge_features)

            # Initialize [t,0] with edge features
            edge_feature_store[:, 0 : self.message_feature_dim] = edge_features_enc

            # Propagate messages
            for prop_idx in range(1, self.n_prop_update_steps + 1) :
                nf_idx = self.message_feature_dim * prop_idx
                interaction_features = edge_feature_store[:, nf_idx - self.message_feature_dim : nf_idx]

                node_features_enc, messages = self.prop_layer(node_features_enc, from_idx, to_idx, interaction_features, return_msg=True)
                updated_edge_feature_store[:, nf_idx : nf_idx + self.message_feature_dim] = torch.clone(messages)

            edge_feature_store = torch.clone(updated_edge_feature_store)

            edge_feature_store_split = torch.split(edge_feature_store, edge_counts.tolist(), dim=0)
            edge_feature_store_query = edge_feature_store_split[0::2]
            edge_feature_store_corpus = edge_feature_store_split[1::2]

            stacked_qedge_store_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) for x in edge_feature_store_query])
            stacked_cedge_store_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) for x in edge_feature_store_corpus])
            
            # Compute transport plan
            stacked_qedge_final_emb = stacked_qedge_store_emb[:,:,-self.message_feature_dim:]
            stacked_cedge_final_emb = stacked_cedge_store_emb[:,:,-self.message_feature_dim:]
            transformed_qedge_final_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qedge_final_emb)))
            transformed_cedge_final_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cedge_final_emb)))
            
            qgraph_mask = torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_sizes])
            cgraph_mask = torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_sizes])
            masked_qedge_final_emb = torch.mul(qgraph_mask,transformed_qedge_final_emb)
            masked_cedge_final_emb = torch.mul(cgraph_mask,transformed_cedge_final_emb)

            sinkhorn_input = torch.matmul(masked_qedge_final_emb, masked_cedge_final_emb.permute(0, 2, 1))
            transport_plan = pytorch_sinkhorn_iters(self.av, sinkhorn_input)

            # Compute interaction
            qedge_features_from_cedges = torch.bmm(transport_plan, stacked_cedge_store_emb)
            cedge_features_from_qedges = torch.bmm(transport_plan.permute(0, 2, 1), stacked_qedge_store_emb)
            interleaved_edge_features = torch.cat([
                qedge_features_from_cedges.unsqueeze(1),
                cedge_features_from_qedges.unsqueeze(1)
            ], dim=1)[:, :, :, self.message_feature_dim:].reshape(-1, self.n_prop_update_steps * self.message_feature_dim) 
            edge_feature_store[:, self.message_feature_dim:] = interleaved_edge_features[edge_indices, :]

        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum(torch.maximum(
            stacked_qedge_final_emb - transport_plan@stacked_cedge_final_emb,
            torch.tensor([0], device=self.device)), dim=(1,2)
        )
        
        return scores