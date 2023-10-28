import torch
import torch.nn.functional as F
from subgraph.utils import cudavar
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen

class EdgeEarlyInteraction(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
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

    def build_layers(self):

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        
        self.fc_combine_interaction = torch.nn.Sequential(
            torch.nn.Linear(prop_config['node_state_dim'] + 1, 1 + prop_config['node_state_dim']),
            torch.nn.ReLU(),
            torch.nn.Linear(1 + prop_config['node_state_dim'], prop_config['node_state_dim'])
        )

        #NOTE:FILTERS_3 is 10 for now - hardcoded into config
        self.fc_transform1 = torch.nn.Linear(2*self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        
        #self.edge_score_fc = torch.nn.Linear(self.prop_layer._message_net[-1].out_features, 1)
        
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
        #a,b = zip(*batch_data_sizes)
        #qgraph_sizes = cudavar(self.av,torch.tensor(a))
        #cgraph_sizes = cudavar(self.av,torch.tensor(b))
        #A
        #a, b = zip(*batch_adj)
        #q_adj = torch.stack(a)
        #c_adj = torch.stack(b)
        
        qgraph_sizes, cgraph_sizes = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av, torch.tensor(qgraph_sizes))
        cgraph_sizes = cudavar(self.av, torch.tensor(cgraph_sizes))
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]
        batch_data_sizes_flat_tensor = cudavar(self.av, torch.LongTensor(batch_data_sizes_flat))
        cumulative_sizes = torch.cumsum(batch_data_sizes_flat_tensor, dim=0)

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)

        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        num_edges, edge_feature_dim = encoded_edge_features.shape
        num_nodes, node_feature_dim = encoded_node_features.shape
        batch_size = len(batch_data_sizes)

        n_time_update_steps = self.config['early_interaction']['n_time_updates']
        n_prop_update_steps = self.config['graph_embedding_net']['n_prop_layers']

        edge_feature_store = torch.zeros(num_nodes, edge_feature_dim * (n_prop_update_steps + 1), device=edge_features.device)
        updated_edge_feature_store = torch.zeros_like(edge_feature_store)
        
        max_set_size_arange = cudavar(self.av, torch.arange(self.max_set_size, dtype=torch.long).reshape(1, -1).repeat(batch_size * 2, 1))
        edge_presence_mask = max_set_size_arange < batch_data_sizes_flat_tensor.unsqueeze(1)
        max_set_size_arange[1:, ] += cumulative_sizes[:-1].unsqueeze(1)
        edge_indices = max_set_size_arange[edge_presence_mask]
        
        for time_idx in range(1, n_time_update_steps + 1):
            node_features_enc = torch.clone(encoded_node_features)
            edge_features_enc = torch.clone(encoded_edge_features)
            for prop_idx in range(1, n_prop_update_steps + 1) :
                nf_idx = edge_feature_dim * prop_idx
                if self.config['early_interaction']['time_update_idx'] == "k_t":
                    interaction_features = edge_feature_store[:, nf_idx - edge_feature_dim : nf_idx]
                elif False:#self.config['early_interaction']['time_update_idx'] == "kp1_t":
                    interaction_features = node_feature_store[:, nf_idx : nf_idx + node_feature_dim]
                
                combined_features = self.fc_combine_interaction(torch.cat([node_features_enc, interaction_features], dim=1))
                if False:#self.config['early_interaction']['prop_separate_params']:
                    node_features_enc = self.prop_layers[time_idx - 1](combined_features, from_idx, to_idx, edge_features_enc)
                else:
                    node_features_enc = self.prop_layer(combined_features, from_idx, to_idx, edge_features_enc)
                # updated_edge_feature_store[:, nf_idx : nf_idx + edge_feature_dim] = torch.clone(node_features_enc)

            # node_feature_store = torch.clone(updated_node_feature_store)

            source_node_enc = node_features_enc[from_idx]
            dest_node_enc  = node_features_enc[to_idx]
            forward_edge_input = torch.cat((source_node_enc,dest_node_enc,edge_features_enc),dim=-1)
            backward_edge_input = torch.cat((dest_node_enc,source_node_enc,edge_features_enc),dim=-1)
            forward_edge_msg = self.prop_layer._message_net(forward_edge_input)
            backward_edge_msg = self.prop_layer._reverse_message_net(backward_edge_input)
            edge_features_enc = forward_edge_msg + backward_edge_msg
            
            edge_counts  = self.fetch_edge_counts(to_idx,from_idx,graph_idx,2*len(batch_data_sizes))
            qgraph_edge_sizes = cudavar(self.av,torch.tensor(edge_counts[0::2]))
            cgraph_edge_sizes = cudavar(self.av,torch.tensor(edge_counts[1::2]))

            edge_feature_enc_split = torch.split(edge_features_enc, edge_counts, dim=0)
            edge_feature_enc_query = edge_feature_enc_split[0::2]
            edge_feature_enc_corpus = edge_feature_enc_split[1::2]  
            
            
            stacked_qedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                            for x in edge_feature_enc_query])
            stacked_cedge_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                            for x in edge_feature_enc_corpus])

            transformed_qedge_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qedge_emb)))
            transformed_cedge_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cedge_emb)))
            qgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_edge_sizes]))
            cgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_edge_sizes]))
            masked_qedge_emb = torch.mul(qgraph_mask,transformed_qedge_emb)
            masked_cedge_emb = torch.mul(cgraph_mask,transformed_cedge_emb)
    
            sinkhorn_input = torch.matmul(masked_qedge_emb,masked_cedge_emb.permute(0,2,1))
            transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input)

            qnode_features_from_cedges = torch.bmm(transport_plan, stacked_cedge_emb)
            cnode_features_from_qedges = torch.bmm(transport_plan.permute(0, 2, 1), stacked_qedge_emb)
            interleaved_node_features = torch.cat([
                qnode_features_from_cedges.unsqueeze(1),
                cnode_features_from_qedges.unsqueeze(1)
            ], dim=1).permute(0, 2, 1, 3).reshape(-1, n_prop_update_steps * edge_feature_dim)
            edge_feature_store[:, edge_feature_dim:] = interleaved_node_features[edge_indices, :]

 
        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum(torch.maximum(stacked_qedge_emb - transport_plan@stacked_cedge_emb,\
              cudavar(self.av,torch.tensor([0]))),\
           dim=(1,2))
        
        return scores
