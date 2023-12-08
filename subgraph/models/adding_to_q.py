import torch
import torch.nn.functional as F
from subgraph.utils import cudavar
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen

class AddingToQ(torch.nn.Module):
    def __init__(self, av, config, input_dim):
        super(AddingToQ, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
    
    def build_masking_utility(self):
        self.max_edge_size = self.av.MAX_EDGES
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
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        
        self.fc_combine_interaction = torch.nn.Sequential(
            torch.nn.Linear(2 * prop_config['node_state_dim'], 2 * prop_config['node_state_dim']),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * prop_config['node_state_dim'], prop_config['node_state_dim'])
        )
        self.fc_transform1 = torch.nn.Linear(self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
    
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

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)

        # adding (q->q_pad)
        mask_from_idx = torch.ones(len(from_idx))
        new_nodes = torch.cat([self.max_set_size * idx * 2 + torch.arange(self.max_set_size)[batch_data_sizes[idx][0]:batch_data_sizes[idx][1]] for idx in range(len(batch_data_sizes))])
        new_edges_from_idx = torch.cat([(self.max_set_size * idx * 2 + torch.arange(batch_data_sizes[idx][0]).reshape(-1, 1).repeat(1, batch_data_sizes[idx][1] - batch_data_sizes[idx][0]).flatten()) for idx in range(len(batch_data_sizes))])
        new_edges_to_idx = torch.cat([(self.max_set_size * idx * 2 + torch.arange(self.max_set_size)[batch_data_sizes[idx][0]:batch_data_sizes[idx][1]].repeat(batch_data_sizes[idx][0])) for idx in range(len(batch_data_sizes))])
        new_graph_idx = [2*idx for idx in range(len(batch_data_sizes)) for _ in range((batch_data_sizes[idx][1] - batch_data_sizes[idx][0]))]
        node_features = torch.ones(node_features.shape[0] + len(new_nodes)).reshape(-1, 1).to(node_features.device)
        edge_features = torch.ones(edge_features.shape[0] + len(new_edges_from_idx)).reshape(-1, 1).to(node_features.device)
        from_idx = torch.cat((from_idx, new_edges_from_idx.to(node_features.device)))
        to_idx = torch.cat((to_idx, new_edges_to_idx.to(node_features.device)))
        graph_idx = torch.cat((graph_idx, torch.tensor(new_graph_idx).to(node_features.device)))

        mask_from_idx = torch.cat((mask_from_idx, torch.zeros(len(from_idx) - len(mask_from_idx)))).to(node_features.device)

        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)

        n_time_update_steps = self.config['early_interaction']['n_time_updates']
        n_prop_update_steps = self.config['graph_embedding_net']['n_prop_layers']

        for time_idx in range(1, n_time_update_steps + 1):
            node_features_enc = torch.clone(encoded_node_features)
            edge_features_enc = torch.clone(encoded_edge_features)
            for prop_idx in range(1, n_prop_update_steps + 1) :
                node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, edge_features_enc, mask_from_idx=mask_from_idx)

            node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
            node_feature_enc_query = node_feature_enc_split[0::2]
            node_feature_enc_corpus = node_feature_enc_split[1::2]

            stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                            for x in node_feature_enc_query])
            stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                            for x in node_feature_enc_corpus])

            transformed_qnode_final_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qnode_emb)))
            transformed_cnode_final_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cnode_emb)))
            
            qgraph_mask = cudavar(self.av, torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_sizes]))
            cgraph_mask = cudavar(self.av, torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_sizes]))
            masked_qnode_final_emb = torch.mul(qgraph_mask,transformed_qnode_final_emb)
            masked_cnode_final_emb = torch.mul(cgraph_mask,transformed_cnode_final_emb)

            sinkhorn_input = torch.matmul(masked_qnode_final_emb, masked_cnode_final_emb.permute(0, 2, 1))
            transport_plan = pytorch_sinkhorn_iters(self.av, sinkhorn_input)

            
            ## Compute Special Graph
            # obtain the to and fro of edges in corpus
            corpus_to_idx = torch.cat(torch.split(to_idx, edge_counts, dim=0)[1::2])
            corpus_from_idx = torch.cat(torch.split(from_idx, edge_counts, dim=0)[1::2])
            # obtain the offsets for each of the edge vertices above
            # (15 .. 15 45 .. 45 75 .. 75)
            offsets = (torch.arange(2 * len(batch_data_sizes))[1::2] * self.max_edge_size).reshape(-1, 1).repeat(1, self.max_edge_size).flatten()
            # (15) * edges_1 (45) * edges_2
            offsets = torch.cat(torch.split(offsets, torch.stack((torch.tensor(edge_counts[1::2]), self.max_edge_size - torch.tensor(edge_counts[1::2])), dim=1).flatten().tolist())[::2], dim=0)
            offsets = offsets.to(corpus_to_idx.device)
            # subtract offset
            corpus_to_idx -= offsets
            corpus_from_idx -= offsets
            print(transport_plan.shape)
            print(corpus_to_idx.shape)

        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum(torch.maximum(
            stacked_qnode_emb - transport_plan@stacked_cnode_emb,
            cudavar(self.av,torch.tensor([0]))),
           dim=(1,2))
        
        return scores