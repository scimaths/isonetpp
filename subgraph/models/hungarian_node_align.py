import torch
import numpy as np
import torch.nn.functional as F
from subgraph.utils import cudavar, adjacency_matrix_from_batched_data, plot_permuted_graphs
from subgraph.models.utils import pytorch_sinkhorn_iters
import GMN.graphembeddingnetwork as gmngen
from subgraph.models.node_align_node_loss import Node_align_Node_loss
from scipy.optimize import linear_sum_assignment

class Hungarian_Node_align_Node_loss(torch.nn.Module):
    def __init__(self, av, config, input_dim):
        super(Hungarian_Node_align_Node_loss, self).__init__()
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
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        
        # Asymmetric network
        self.asymm_encoder = gmngen.GraphEncoder(**self.config['encoder'])
        self.asymm_prop_layer = gmngen.GraphPropLayer(**prop_config)
        self.asymm_fc_transform1 = torch.nn.Linear(self.av.filters_3, self.av.transform_dim)
        self.asymm_fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()

        # Symmetric network
        self.symm_encoder = gmngen.GraphEncoder(**self.config['encoder'])
        self.symm_prop_layer = gmngen.GraphPropLayer(**prop_config)
        self.symm_fc_transform1 = torch.nn.Linear(self.av.filters_3, self.av.transform_dim)
        self.symm_fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()

    def embeddings_to_transport_plan(self, node_features_enc, batch_data_sizes, batch_data_sizes_flat, qgraph_sizes, cgraph_sizes, lrl):
        #[(8, 12), (10, 13), (10, 14)] -> [8, 12, 10, 13, 10, 14]
        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]
        assert(list(zip([x.shape[0] for x in node_feature_enc_query], [x.shape[0] for x in node_feature_enc_corpus])) == batch_data_sizes)        
        
        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) for x in node_feature_enc_corpus])

        transformed_qnode_emb = lrl(stacked_qnode_emb)
        transformed_cnode_emb = lrl(stacked_cnode_emb)

        qgraph_mask = cudavar(self.av, torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_sizes]))
        cgraph_mask = cudavar(self.av, torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_sizes]))
        masked_qnode_emb = torch.mul(qgraph_mask, transformed_qnode_emb)
        masked_cnode_emb = torch.mul(cgraph_mask, transformed_cnode_emb)
 
        sinkhorn_input = torch.matmul(masked_qnode_emb, masked_cnode_emb.permute(0,2,1))
        transport_plan = pytorch_sinkhorn_iters(self.av, sinkhorn_input)
        return transport_plan, stacked_qnode_emb, stacked_cnode_emb

    def forward(self, batch_data, batch_data_sizes, batch_adj):
        import time
        c = time.time()

        def flatten(list_of_tuples):
            return [item for sublist in list_of_tuples for item in sublist]
        a, b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))
        batch_data_sizes_flat = flatten(batch_data_sizes)
        batch_data_sizes_flat_tensor = cudavar(self.av,torch.Tensor(batch_data_sizes_flat))

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)

        # Asymmetric network
        node_features_enc, edge_features_enc = self.asymm_encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            node_features_enc = self.asymm_prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)

        asymm_lrl = lambda x: self.asymm_fc_transform2(self.relu1(self.asymm_fc_transform1(x)))
        asymm_transport_plan, asymm_stacked_qnode_emb, asymm_stacked_cnode_emb = self.embeddings_to_transport_plan(
            node_features_enc=node_features_enc, batch_data_sizes=batch_data_sizes, batch_data_sizes_flat=batch_data_sizes_flat,
            qgraph_sizes=qgraph_sizes, cgraph_sizes=cgraph_sizes, lrl=asymm_lrl
        )
        asymm_score = -torch.sum(torch.maximum(asymm_stacked_qnode_emb - asymm_transport_plan@asymm_stacked_cnode_emb, cudavar(self.av,torch.tensor([0]))), dim=(1,2))

        # Deducing induced graphs - Using entire TRANSPORT_PLAN
        # hungarian_matchings = torch.cat([
        #     cudavar(self.av, torch.LongTensor(np.array(linear_sum_assignment(-transport_plan.detach().cpu().numpy()))))
        # for transport_plan in asymm_transport_plan])

        # Deducing induced graphs - Filtering TRANSPORT_PLAN
        hungarian_matchings = []
        for idx in range(len(asymm_transport_plan)):
            qgraph_size, cgraph_size = qgraph_sizes[idx], cgraph_sizes[idx]
            indices = torch.cat([torch.arange(self.max_set_size, dtype=torch.long).unsqueeze(0)] * 2)
            modified_transport_plan = asymm_transport_plan[idx, :cgraph_size, :cgraph_size].detach().cpu().numpy()
            modified_transport_plan[qgraph_size:, :] = 0
            indices[:, :cgraph_size] = torch.LongTensor(np.array(linear_sum_assignment(-modified_transport_plan)))
            hungarian_matchings.append(cudavar(self.av, indices))
        hungarian_matchings = torch.cat(hungarian_matchings)

        induced_batch_data_sizes_flat_tensor = qgraph_sizes.repeat_interleave(2)
        total_nodes_till = cudavar(self.av, torch.zeros(len(batch_data_sizes_flat), 1, dtype=torch.long))
        induced_total_nodes_till = cudavar(self.av, torch.zeros(len(batch_data_sizes_flat), 1, dtype=torch.long))
        total_nodes_till[1:, 0] = torch.cumsum(batch_data_sizes_flat_tensor, dim=0)[:-1]
        induced_total_nodes_till[1:, 0] = torch.cumsum(induced_batch_data_sizes_flat_tensor, dim=0)[:-1]
        max_set_size_arange = cudavar(self.av, torch.arange(self.max_set_size, dtype=torch.long)).unsqueeze(0)

        node_renumberings = cudavar(self.av, torch.zeros(len(node_features), dtype=torch.long))
        original_graph_numbering = hungarian_matchings + total_nodes_till
        new_graph_numbering = max_set_size_arange + induced_total_nodes_till
        non_padding_nodes_mask = hungarian_matchings < batch_data_sizes_flat_tensor.unsqueeze(1)
        node_renumberings[original_graph_numbering[non_padding_nodes_mask]] = new_graph_numbering[non_padding_nodes_mask]

        induced_nodes_mask = cudavar(self.av, torch.zeros(len(node_features), dtype=torch.bool))
        induced_graph_size_mask = max_set_size_arange < induced_batch_data_sizes_flat_tensor.unsqueeze(1)
        induced_nodes_mask[original_graph_numbering[torch.logical_and(non_padding_nodes_mask, induced_graph_size_mask)]] = 1
        edge_included = torch.logical_and(induced_nodes_mask[from_idx], induced_nodes_mask[to_idx])
        induced_from_idx = node_renumberings[from_idx[edge_included]]
        induced_to_idx = node_renumberings[to_idx[edge_included]]

        induced_node_features = cudavar(self.av, torch.ones(size=(torch.sum(qgraph_sizes) * 2, node_features.shape[1])))
        induced_edge_features = edge_features[edge_included, :]
        induced_batch_data_sizes = [(size, size) for size in qgraph_sizes]

        # TESTING
        # print(batch_data_sizes)
        # print("From", from_idx)
        # print("To", to_idx)
        # print("Hungarian", hungarian_matchings)
        # print("Induced from", torch.Tensor(induced_from_idx))
        # print("Induced to", torch.Tensor(induced_to_idx))

        # NOT-VECTORIZED IMPLEMENTATION
        # hungarian_matchings_ = np.array([linear_sum_assignment(-transport_plan.detach().cpu().numpy())[1] for transport_plan in asymm_transport_plan])
        # from_idx = from_idx.detach().cpu().numpy()
        # to_idx = to_idx.detach().cpu().numpy()
        # induced_from_idx = []
        # induced_to_idx = []
        # induced_batch_data_sizes = []
        # edge_idx = 0
        # total_actual_nodes_till = 0
        # total_induced_nodes_till = 0
        # for idx in range(len(batch_data_sizes)):
        #     qgraph_size = qgraph_sizes[idx].item()
        #     cgraph_size = cgraph_sizes[idx].item()
        #     while edge_idx < len(from_idx) and from_idx[edge_idx] < total_actual_nodes_till + qgraph_size:
        #         induced_from_idx.append(from_idx[edge_idx] - total_actual_nodes_till + total_induced_nodes_till)
        #         induced_to_idx.append(to_idx[edge_idx] - total_actual_nodes_till + total_induced_nodes_till)
        #         edge_idx += 1
        #     total_induced_nodes_till += qgraph_size
        #     total_actual_nodes_till += qgraph_size
        #     chosen_cnodes_mask = -np.ones(shape=(self.max_set_size))
        #     chosen_cnodes_mask[hungarian_matchings_[idx][:qgraph_size]] = np.arange(qgraph_size)
        #     while edge_idx < len(from_idx) and from_idx[edge_idx] < total_actual_nodes_till + cgraph_size:
        #         from_offset = chosen_cnodes_mask[from_idx[edge_idx] - total_actual_nodes_till]
        #         to_offset = chosen_cnodes_mask[to_idx[edge_idx] - total_actual_nodes_till]
        #         if from_offset >= 0 and to_offset >= 0:
        #             induced_from_idx.append(from_offset + total_induced_nodes_till)
        #             induced_to_idx.append(to_offset + total_induced_nodes_till)
        #         edge_idx += 1
        #     total_induced_nodes_till += qgraph_size
        #     total_actual_nodes_till += cgraph_size
        #     induced_batch_data_sizes.append((qgraph_size, qgraph_size))
        # induced_from_idx = cudavar(self.av, torch.LongTensor(induced_from_idx))
        # induced_to_idx = cudavar(self.av, torch.LongTensor(induced_to_idx))

        # Symmetric network
        induced_node_features_enc, induced_edge_features_enc = self.symm_encoder(induced_node_features, induced_edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            induced_node_features_enc = self.symm_prop_layer(induced_node_features_enc, induced_from_idx, induced_to_idx, induced_edge_features_enc)

        symm_lrl = lambda x: self.symm_fc_transform2(self.relu1(self.symm_fc_transform1(x)))
        symm_transport_plan, symm_stacked_qnode_emb, symm_stacked_cnode_emb = self.embeddings_to_transport_plan(
            node_features_enc=induced_node_features_enc, batch_data_sizes=induced_batch_data_sizes, batch_data_sizes_flat=flatten(induced_batch_data_sizes),
            qgraph_sizes=qgraph_sizes, cgraph_sizes=qgraph_sizes, lrl=symm_lrl
        )
        symm_score = -torch.sum((symm_stacked_qnode_emb - symm_transport_plan@symm_stacked_cnode_emb) ** 2, dim=(1,2))

        if self.training:
            return asymm_score + symm_score
        else:
            if self.diagnostic:
                return asymm_transport_plan, symm_transport_plan
            else:
                return symm_score

    def visualize(self, vis_idxs, batch_data, batch_data_sizes, batch_adj):
        self.eval()
        self.diagnostic = True
        asymm_plan, symm_plan = self.forward(batch_data, batch_data_sizes, batch_adj)
        _, _, from_idx, to_idx, _ = self.get_graph(batch_data)
        for idx in vis_idxs:
            query_adj, corpus_adj = adjacency_matrix_from_batched_data(idx, batch_data_sizes, from_idx, to_idx, self.max_set_size)
            plot_permuted_graphs(query_adj, corpus_adj, linear_sum_assignment(-asymm_plan[idx].detach().cpu())[1], f"trunc_plan_asymm_{idx}")
            plot_permuted_graphs(query_adj, corpus_adj, linear_sum_assignment(-symm_plan[idx].detach().cpu())[1], f"trunc_plan_symm_{idx}")