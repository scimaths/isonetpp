import torch
import torch.nn.functional as F
from subgraph.utils import cudavar
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen

class EdgeEarlyInteraction(torch.nn.Module):
    def __init__(self, av, config, input_dim):
        super(EdgeEarlyInteraction, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
    
    def build_masking_utility(self):
        self.max_set_size = self.av.MAX_EDGES
        #this mask pattern sets bottom last few rows to 0 based on padding needs
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))) for x in range(0,self.max_set_size+1)]
        # Mask pattern sets top left (k)*(k) square to 1 inside arrays of size n*n. Rest elements are 0
        self.set_size_to_mask_map = [torch.cat((torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([x,self.max_set_size-x])).repeat(x,1),
                             torch.repeat_interleave(torch.tensor([1,0]),torch.tensor([0,self.max_set_size])).repeat(self.max_set_size-x,1)))
                             for x in range(0,self.max_set_size+1)]


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
        self.prop_config = self.config['graph_embedding_net'].copy()
        self.prop_config.pop('n_prop_layers',None)
        self.prop_config.pop('share_prop_params',None)
        self.final_edge_encoding_dim = 30
        self.prop_config['final_edge_encoding_dim'] = self.final_edge_encoding_dim
        self.message_feature_dim = self.prop_config['edge_hidden_sizes'][-1]
        self.prop_layer = gmngen.GraphPropLayer(**self.prop_config)
        
        combined_feature_dim = self.message_feature_dim + self.config['encoder']['edge_feature_dim']
        self.fc_combine_interaction = torch.nn.Sequential(
            torch.nn.Linear(combined_feature_dim, combined_feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(combined_feature_dim, self.final_edge_encoding_dim)
        )
        self.fc_transform1 = torch.nn.Linear(2*self.av.filters_3, self.av.transform_dim)
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
        
        edge_counts  = self.fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
        edge_counts = torch.tensor(edge_counts, device=device, dtype=torch.long)
        
        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        num_edges, edge_feature_dim = encoded_edge_features.shape
        batch_size = len(batch_data_sizes)

        n_time_update_steps = self.config['early_interaction']['n_time_updates']
        n_prop_update_steps = self.config['graph_embedding_net']['n_prop_layers']

        edge_feature_store = torch.zeros(num_edges, self.message_feature_dim * (n_prop_update_steps + 1), device=node_features.device)
        updated_edge_feature_store = torch.zeros_like(edge_feature_store)

        max_set_size_arange = torch.arange(self.max_set_size, dtype=torch.long, device=device).reshape(1, -1).repeat(batch_size * 2, 1)
        edge_presence_mask = max_set_size_arange < edge_counts.unsqueeze(1)
        max_set_size_arange[1:, ] += cumulative_sizes[:-1].unsqueeze(1)
        edge_indices = max_set_size_arange[edge_presence_mask]

        for time_idx in range(1, n_time_update_steps + 1):
            node_features_enc = torch.clone(encoded_node_features)
            edge_features_enc = torch.clone(encoded_edge_features)
            for prop_idx in range(1, n_prop_update_steps + 1) :
                nf_idx = self.message_feature_dim * prop_idx
                interaction_features = edge_feature_store[:, nf_idx - self.message_feature_dim : nf_idx]
                combined_features = self.fc_combine_interaction(torch.cat([edge_features_enc, interaction_features], dim=1))

                node_features_enc_out = self.prop_layer(node_features_enc, from_idx, to_idx, combined_features)

                source_node_enc = node_features_enc[from_idx]
                dest_node_enc  = node_features_enc[to_idx]
                forward_edge_input = torch.cat((source_node_enc,dest_node_enc,combined_features),dim=-1)
                backward_edge_input = torch.cat((dest_node_enc,source_node_enc,combined_features),dim=-1)
                forward_edge_msg = self.prop_layer._message_net(forward_edge_input)
                backward_edge_msg = self.prop_layer._reverse_message_net(backward_edge_input)
                messages = forward_edge_msg + backward_edge_msg
                
                node_features_enc = node_features_enc_out
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
            
            qgraph_mask = cudavar(self.av, torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_sizes]))
            cgraph_mask = cudavar(self.av, torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_sizes]))
            masked_qedge_final_emb = torch.mul(qgraph_mask,transformed_qedge_final_emb)
            masked_cedge_final_emb = torch.mul(cgraph_mask,transformed_cedge_final_emb)

            sinkhorn_input = torch.matmul(masked_qedge_final_emb, masked_cedge_final_emb.permute(0, 2, 1))
            transport_plan = pytorch_sinkhorn_iters(self.av, sinkhorn_input)

            # Compute interaction
            qnode_features_from_cnodes = torch.bmm(transport_plan, stacked_cedge_store_emb)
            cnode_features_from_qnodes = torch.bmm(transport_plan.permute(0, 2, 1), stacked_qedge_store_emb)
            interleaved_node_features = torch.cat([
                qnode_features_from_cnodes.unsqueeze(1),
                cnode_features_from_qnodes.unsqueeze(1)
            ], dim=1)[:, :, :, self.message_feature_dim:].reshape(-1, n_prop_update_steps * self.message_feature_dim) 
            edge_feature_store[:, self.message_feature_dim:] = interleaved_node_features[edge_indices, :]

        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum(torch.maximum(
            stacked_qedge_final_emb - transport_plan@stacked_cedge_final_emb,
            cudavar(self.av,torch.tensor([0]))),
           dim=(1,2))
        
        return scores
