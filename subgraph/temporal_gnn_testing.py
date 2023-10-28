import scipy
import torch
import argparse
import colorsys
import numpy as np
import networkx as nx
from GMN.configure import *
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from subgraph.utils import cudavar
import GMN.graphembeddingnetwork as gmngen
from subgraph.earlystopping import EarlyStoppingModule
from subgraph.models.utils import pytorch_sinkhorn_iters
from subgraph.iso_matching_models import OurMatchingModelSubgraphIsoData, seed_everything

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

total_count = 0
mismatch_count = 0

class Node_align_Node_loss(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(Node_align_Node_loss, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False
        
    def build_masking_utility(self):
        self.max_set_size = max(self.av.MAX_QUERY_SUBGRAPH_SIZE,self.av.MAX_CORPUS_SUBGRAPH_SIZE)
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1,self.av.transform_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1,self.av.transform_dim))).cuda() for x in range(0,self.max_set_size+1)]
        

    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers',None)
        prop_config.pop('share_prop_params',None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        
        #NOTE:FILTERS_3 is 10 for now - hardcoded into config
        self.fc_transform1 = torch.nn.Linear(self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.av.transform_dim, self.av.transform_dim)
        
    def get_graph(self, batch):
        graph = batch
        node_features = cudavar(self.av,torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av,torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av,torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av,torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av,torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx    

    def visualize(self, index, transport_plan, batch_data_sizes, to_idx, from_idx):
        # (q, c)
        transport_plan = transport_plan[index].detach().to('cpu').numpy()
        from_idx_dupl = from_idx.detach().to('cpu').numpy()
        to_idx_dupl = to_idx.detach().to('cpu').numpy()
        batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
        
        # logs
        # print("Transport plan", np.array_str(transport_plan, precision=3, suppress_small=False))
        # print("Row sum of transport plan", transport_plan.sum(axis=1))
        # print("Column sum of transport plan", transport_plan.sum(axis=0))
        print("Shape of transport plan", transport_plan.shape)
        row_argmax = np.argmax(transport_plan, 1)
        print("Row argmax of transport plan", row_argmax)
        print("Graph sizes", batch_data_sizes[index])
        q_size = batch_data_sizes[index][0]
        c_size = batch_data_sizes[index][1]

        hung_q, hung_c = scipy.optimize.linear_sum_assignment(-transport_plan)
        hung_alignment = np.arange(15)
        print("Hungarian indices")
        print(hung_q)
        print(hung_c)
        hung_alignment[hung_c] = hung_q
        if c_size < 15:
            # if any corpus padding nodes is being mapped to query nodes
            # True if it is
            hung_mismatch = np.min(hung_alignment[c_size:]) < q_size
        else:
            hung_mismatch = 0
        print("Flipped Hungarian Alignment", hung_alignment, hung_mismatch)

        # design the permutation matrix
        permutation = np.argmax(transport_plan, axis=1)
        permutation_matrix = np.zeros((self.max_set_size, self.max_set_size))
        for i in range(self.max_set_size):
            permutation_matrix[i, permutation[i]] = 1

        def choose_colors(num_colors):
            np.random.seed(42)
            colors=[]
            for i in np.arange(0., 360., 360. / num_colors):
                hue = i/360.
                lightness = (30 + np.random.rand() * 70)/100.0
                saturation = (30 + np.random.rand() * 70)/100.0
                colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
            return np.array(colors)

        set_col = choose_colors(self.max_set_size)
        def get(index_qc):
            if index_qc:
                min_num = sum(batch_data_sizes_flat[ : 2 * index + 1])
                max_num = sum(batch_data_sizes_flat[ : 2 * index + 2])
                indexing = hung_q
            else:
                min_num = sum(batch_data_sizes_flat[ : 2 * index ])
                max_num = sum(batch_data_sizes_flat[ : 2 * index + 1])
                indexing = hung_c
            print("Visualizing min_num", min_num, "max_num", max_num)
        
            # adjacency_matrix = [[0 for i in range(batch_data_sizes[index][index_qc])] for j in range(batch_data_sizes[index][index_qc])]
            adjacency_matrix = [[0 for i in range(self.max_set_size)] for j in range(self.max_set_size)]
        
            for i in range(len(from_idx_dupl)):
                if from_idx_dupl[i] >= min_num and from_idx_dupl[i] < max_num:
                    adjacency_matrix[from_idx_dupl[i] - min_num][to_idx_dupl[i] - min_num] = 1
                    adjacency_matrix[to_idx_dupl[i] - min_num][from_idx_dupl[i] - min_num] = 1
                    print("Edge -", from_idx_dupl[i], to_idx_dupl[i])
        
            adjacency_matrix = np.array(adjacency_matrix)
        
            # if index_qc:
            #     adjacency_matrix = permutation_matrix @ adjacency_matrix @ permutation_matrix.transpose()
            #     num_nodes_q = batch_data_sizes[index][0]
            #     num_nodes_c = batch_data_sizes[index][1]
            #     transport_plan_here = transport_plan[index].detach().to('cpu').numpy()
            #     argmax_indices = np.argmax(transport_plan_here, axis=1)
            #     filter_indices = argmax_indices[:num_nodes_q]
            #     print(filter_indices)
            #     adjacency_matrix = adjacency_matrix[filter_indices, :][:, filter_indices]
            from networkx.drawing.nx_agraph import graphviz_layout

            graph = nx.from_numpy_array(adjacency_matrix)
            layout = graphviz_layout(graph, prog='neato')  # You can try different layouts here
            nx.draw(graph, pos = layout, with_labels=True, node_color=set_col[indexing])
            loc = 'index_node_align_unaligned_' + str(index) + '_' + str(index_qc) + '.png'
            print(loc)
            plt.savefig(loc)
            plt.clf()

        if not hung_mismatch:
            get(0)
            get(1)
            # input()
        return hung_mismatch
        
    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
        """
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))
        #A
        a, b = zip(*batch_adj)
        q_adj = torch.stack(a)
        c_adj = torch.stack(b)

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
    
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config['graph_embedding_net'] ['n_prop_layers']) :
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)

        #[(8, 12), (10, 13), (10, 14)] -> [8, 12, 10, 13, 10, 14]
        batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
        node_feature_enc_split = torch.split(node_features_enc, batch_data_sizes_flat, dim=0)
        node_feature_enc_query = node_feature_enc_split[0::2]
        node_feature_enc_corpus = node_feature_enc_split[1::2]
        assert(list(zip([x.shape[0] for x in node_feature_enc_query], \
                        [x.shape[0] for x in node_feature_enc_corpus])) \
               == batch_data_sizes)        
        
        stacked_qnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in node_feature_enc_query])
        stacked_cnode_emb = torch.stack([F.pad(x, pad=(0,0,0,self.max_set_size-x.shape[0])) \
                                         for x in node_feature_enc_corpus])

        transformed_qnode_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_qnode_emb)))
        transformed_cnode_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_cnode_emb)))
        qgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in qgraph_sizes]))
        cgraph_mask = cudavar(self.av,torch.stack([self.graph_size_to_mask_map[i] for i in cgraph_sizes]))
        masked_qnode_emb = torch.mul(qgraph_mask,transformed_qnode_emb)
        masked_cnode_emb = torch.mul(cgraph_mask,transformed_cnode_emb)
 
        sinkhorn_input = torch.matmul(masked_qnode_emb,masked_cnode_emb.permute(0,2,1))
        # transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input, n_iters=2000)
        transport_plan = pytorch_sinkhorn_iters(self.av,sinkhorn_input, n_iters=20)

        if self.diagnostic_mode:
            return transport_plan
        
        scores = -torch.sum(torch.maximum(stacked_qnode_emb - transport_plan@stacked_cnode_emb,\
              cudavar(self.av,torch.tensor([0]))),\
           dim=(1,2))

        mismatches = 0
        total = 0

        # visualize
        scores = scores.detach().to('cpu').numpy()
        print(scores.shape)
        indices_descending = np.argsort(scores)[::-1]
        for index in indices_descending:
            total += 1
            print("Scores", scores[index])
            mismatches += self.visualize(index, transport_plan, batch_data_sizes, to_idx, from_idx)

        return mismatches, total



class TemporalGNN(torch.nn.Module):
    def __init__(self, av, config, input_dim):
        super(TemporalGNN, self).__init__()
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
        if self.config['temporal_gnn']['prop_separate_params']:
            self.prop_layers = torch.nn.ModuleList([gmngen.GraphPropLayer(**prop_config) for _ in range(self.config['temporal_gnn']['n_time_updates'])])
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


    def check_change(self, t1, t2, index):
        t1 = t1[index].detach().to('cpu').numpy()
        t2 = t2[index].detach().to('cpu').numpy()
        hung_q1, hung_c1 = scipy.optimize.linear_sum_assignment(-t1)
        hung_q2, hung_c2 = scipy.optimize.linear_sum_assignment(-t2)
        hung_alignment1 = np.arange(15)
        hung_alignment2 = np.arange(15)
        # print("Hungarian indices")
        # print(hung_q)
        # print(hung_c)
        
        hung_alignment1[hung_c1] = hung_q1
        hung_alignment2[hung_c2] = hung_q2
        hung_mismatch = 0
        for i in range(15):
            if hung_alignment1[i] != hung_alignment2[i]:
                hung_mismatch = 1
                print("corpus node", i, "changed mapping from", hung_alignment1[i], "to", hung_alignment2[i])
        return hung_mismatch
        
        
    def visualize(self, index, transport_plan, batch_data_sizes, to_idx, from_idx, t_index):
        # (q, c)
        transport_plan = transport_plan[index].detach().to('cpu').numpy()
        from_idx_dupl = from_idx.detach().to('cpu').numpy()
        to_idx_dupl = to_idx.detach().to('cpu').numpy()
        batch_data_sizes_flat  = [item for sublist in batch_data_sizes for item in sublist]
        
        # logs
        # print("Transport plan", np.array_str(transport_plan, precision=3, suppress_small=False))
        # print("Row sum of transport plan", transport_plan.sum(axis=1))
        # print("Column sum of transport plan", transport_plan.sum(axis=0))
        # print("Shape of transport plan", transport_plan.shape)
        row_argmax = np.argmax(transport_plan, 1)
        # print("Row argmax of transport plan", row_argmax)
        # print("Graph sizes", batch_data_sizes[index])
        q_size = batch_data_sizes[index][0]
        c_size = batch_data_sizes[index][1]

        hung_q, hung_c = scipy.optimize.linear_sum_assignment(-transport_plan)
        hung_alignment = np.arange(15)
        # print("Hungarian indices")
        # print(hung_q)
        # print(hung_c)
        hung_alignment[hung_c] = hung_q
        if c_size < 15:
            # if any corpus padding nodes is being mapped to query nodes
            # True if it is
            hung_mismatch = np.min(hung_alignment[c_size:]) < q_size
        else:
            hung_mismatch = 0
        # print("Flipped Hungarian Alignment", hung_alignment, hung_mismatch)

        # design the permutation matrix
        permutation = np.argmax(transport_plan, axis=1)
        permutation_matrix = np.zeros((self.max_set_size, self.max_set_size))
        for i in range(self.max_set_size):
            permutation_matrix[i, permutation[i]] = 1

        def choose_colors(num_colors):
            np.random.seed(42)
            colors=[]
            for i in np.arange(0., 360., 360. / num_colors):
                hue = i/360.
                lightness = (30 + np.random.rand() * 70)/100.0
                saturation = (30 + np.random.rand() * 70)/100.0
                colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
            return np.array(colors)

        set_col = choose_colors(self.max_set_size)
        def get(index_qc):
            if index_qc:
                min_num = sum(batch_data_sizes_flat[ : 2 * index + 1])
                max_num = sum(batch_data_sizes_flat[ : 2 * index + 2])
                indexing = hung_q
            else:
                min_num = sum(batch_data_sizes_flat[ : 2 * index ])
                max_num = sum(batch_data_sizes_flat[ : 2 * index + 1])
                indexing = hung_c
            # print("Visualizing min_num", min_num, "max_num", max_num)
        
            # adjacency_matrix = [[0 for i in range(batch_data_sizes[index][index_qc])] for j in range(batch_data_sizes[index][index_qc])]
            adjacency_matrix = [[0 for i in range(self.max_set_size)] for j in range(self.max_set_size)]
        
            for i in range(len(from_idx_dupl)):
                if from_idx_dupl[i] >= min_num and from_idx_dupl[i] < max_num:
                    adjacency_matrix[from_idx_dupl[i] - min_num][to_idx_dupl[i] - min_num] = 1
                    adjacency_matrix[to_idx_dupl[i] - min_num][from_idx_dupl[i] - min_num] = 1
                    # print("Edge -", from_idx_dupl[i], to_idx_dupl[i])
        
            adjacency_matrix = np.array(adjacency_matrix)
        
            # if index_qc:
            #     adjacency_matrix = permutation_matrix @ adjacency_matrix @ permutation_matrix.transpose()
            #     num_nodes_q = batch_data_sizes[index][0]
            #     num_nodes_c = batch_data_sizes[index][1]
            #     transport_plan_here = transport_plan[index].detach().to('cpu').numpy()
            #     argmax_indices = np.argmax(transport_plan_here, axis=1)
            #     filter_indices = argmax_indices[:num_nodes_q]
            #     print(filter_indices)
            #     adjacency_matrix = adjacency_matrix[filter_indices, :][:, filter_indices]
            from networkx.drawing.nx_agraph import graphviz_layout

            graph = nx.from_numpy_array(adjacency_matrix)
            layout = graphviz_layout(graph, prog='neato')  # You can try different layouts here
            nx.draw(graph, pos = layout, with_labels=True, node_color=set_col[indexing])
            loc = 'index_node_align_unaligned_' + str(index) + '_' + str(t_index) + '_' + str(index_qc) + '.png'
            # print(loc)
            # plt.savefig(loc)
            # plt.clf()

        if not hung_mismatch:
            get(0)
            get(1)
            # input()
        return hung_mismatch

    def forward(self, batch_data, batch_data_sizes, batch_adj):
        qgraph_sizes, cgraph_sizes = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av, torch.tensor(qgraph_sizes))
        cgraph_sizes = cudavar(self.av, torch.tensor(cgraph_sizes))
        batch_data_sizes_flat = [item for sublist in batch_data_sizes for item in sublist]
        batch_data_sizes_flat_tensor = cudavar(self.av, torch.LongTensor(batch_data_sizes_flat))
        cumulative_sizes = torch.cumsum(batch_data_sizes_flat_tensor, dim=0)

        node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(batch_data)
        
        encoded_node_features, encoded_edge_features = self.encoder(node_features, edge_features)
        num_nodes, node_feature_dim = encoded_node_features.shape
        batch_size = len(batch_data_sizes)

        n_time_update_steps = self.config['temporal_gnn']['n_time_updates']
        n_prop_update_steps = self.config['graph_embedding_net']['n_prop_layers']

        node_feature_store = torch.zeros(num_nodes, node_feature_dim * (n_prop_update_steps + 1), device=node_features.device)
        updated_node_feature_store = torch.zeros_like(node_feature_store)

        max_set_size_arange = cudavar(self.av, torch.arange(self.max_set_size, dtype=torch.long).reshape(1, -1).repeat(batch_size * 2, 1))
        node_presence_mask = max_set_size_arange < batch_data_sizes_flat_tensor.unsqueeze(1)
        max_set_size_arange[1:, ] += cumulative_sizes[:-1].unsqueeze(1)
        node_indices = max_set_size_arange[node_presence_mask]
        transport_plan_list = []
        for time_idx in range(1, n_time_update_steps + 1):
            node_features_enc = torch.clone(encoded_node_features)
            edge_features_enc = torch.clone(encoded_edge_features)
            for prop_idx in range(1, n_prop_update_steps + 1) :
                nf_idx = node_feature_dim * prop_idx
                if self.config['temporal_gnn']['time_update_idx'] == "k_t":
                    interaction_features = node_feature_store[:, nf_idx - node_feature_dim : nf_idx]
                elif self.config['temporal_gnn']['time_update_idx'] == "kp1_t":
                    interaction_features = node_feature_store[:, nf_idx : nf_idx + node_feature_dim]

                combined_features = self.fc_combine_interaction(torch.cat([node_features_enc, interaction_features], dim=1))
                if self.config['temporal_gnn']['prop_separate_params']:
                    node_features_enc = self.prop_layers[time_idx - 1](combined_features, from_idx, to_idx, edge_features_enc)
                else:
                    node_features_enc = self.prop_layer(combined_features, from_idx, to_idx, edge_features_enc)
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
            ], dim=1).permute(0, 2, 1, 3).reshape(-1, n_prop_update_steps * node_feature_dim)
            node_feature_store[:, node_feature_dim:] = interleaved_node_features[node_indices, :]


            scores = -torch.sum(torch.maximum(
                stacked_qnode_final_emb - transport_plan@stacked_cnode_final_emb,
                cudavar(self.av,torch.tensor([0]))),
                dim=(1,2))
            scores = scores.detach().to('cpu').numpy()
            # print(scores.shape)
            indices_descending = np.argsort(scores)[::-1]
        
            for index in indices_descending:
                if self.check_change(transport_plan_list[-1], transport_plan):
                        
            transport_plan_list.append(transport_plan)


        if self.diagnostic_mode:
            return transport_plan

        scores = -torch.sum(torch.maximum(
            stacked_qnode_final_emb - transport_plan@stacked_cnode_final_emb,
            cudavar(self.av,torch.tensor([0]))),
           dim=(1,2))
        mismatches = [0 for _ in range(n_time_update_steps)]
        total = [0 for _ in range(n_time_update_steps)]

        # visualize
        scores = scores.detach().to('cpu').numpy()
        # print(scores.shape)
        indices_descending = np.argsort(scores)[::-1]
        for index in indices_descending[:10]:
            # print("Combing through processing of index", index)
            # print("Scores", scores[index])
            for t_index in range(n_time_update_steps):
                # print("Ashwin")
                total[t_index] += 1
                mismatches[t_index] += self.visualize(index, transport_plan_list[t_index], batch_data_sizes, to_idx, from_idx, t_index)
        print('frog', mismatches, total)
        input()
        return mismatches, total


def test_node_align_node_loss(av, config):
    device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
    train_data = OurMatchingModelSubgraphIsoData(av,mode="train")
    av.MAX_EDGES = max(max([g.number_of_edges() for g in train_data.query_graphs]),\
                    max([g.number_of_edges() for g in train_data.corpus_graphs]))

    es = EarlyStoppingModule(av)

    # model = Fringed_node_align_Node_loss(av, config, 1)
    # model = ISONET(av, config, 1).to(device)
    model = TemporalGNN(av, config, 1).to(device)
    # model = Node_align_Node_loss(av, config, 1).to(device)
    model.load_state_dict(es.load_best_model()['model_state_dict'])
    model.to(device)

    train_data.data_type = "gmn"

    n_batches = train_data.create_stratified_batches()

    total = 0
    mismatches = 0
    n_time_update_steps = config['temporal_gnn']['n_time_updates']
    print(train_data.num_batches)
    input()
    mismatches = [0 for _ in range(n_time_update_steps)]
    total = [0 for _ in range(n_time_update_steps)]
    for batch_idx in range(train_data.num_batches):
        batch_data, batch_data_sizes, _, batch_adj = train_data.fetch_batched_data_by_id(batch_idx)
        m, t = model(batch_data,batch_data_sizes,batch_adj)
        for j in range(n_time_update_steps):
            mismatches[j] += m[j]
            total[j] += t[j]
        print("Batch-IDX", batch_idx, mismatches, total)

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--logpath",                        type=str,   default="logDir/logfile",help="/path/to/log")
  ap.add_argument("--want_cuda",                      type=bool,  default=True)
  ap.add_argument("--RUN_TILL_ES",                    type=bool,  default=True)
  ap.add_argument("--has_cuda",                       type=bool,  default=torch.cuda.is_available())
  ap.add_argument("--is_sig",                         type=bool,  default=False)
  ap.add_argument("--ES",                             type=int,   default=50)
  ap.add_argument("--MIN_QUERY_SUBGRAPH_SIZE",        type=int,   default=5)
  ap.add_argument("--MAX_QUERY_SUBGRAPH_SIZE",        type=int,   default=10)
  ap.add_argument("--MIN_CORPUS_SUBGRAPH_SIZE",       type=int,   default=11)
  ap.add_argument("--MAX_CORPUS_SUBGRAPH_SIZE",       type=int,   default=15)
  ap.add_argument("--MAX_GRAPH_SIZE",                 type=int,   default=0)
  ap.add_argument("--n_layers",                       type=int,   default=3)
  ap.add_argument("--conv_type",                      type=str,   default='SAGE')
  ap.add_argument("--method_type",                    type=str,   default='order')
  ap.add_argument("--skip",                           type=str,   default='learnable')
  ap.add_argument("--neuromatch_hidden_dim",          type=int,   default=10)
  ap.add_argument("--post_mp_dim",                    type=int,   default=64)
  ap.add_argument("--filters_1",                      type=int,   default=128)
  ap.add_argument("--filters_2",                      type=int,   default=64)
  ap.add_argument("--filters_3",                      type=int,   default=10)
  ap.add_argument("--dropout",                        type=float, default=0)
  ap.add_argument("--tensor_neurons",                 type=int,   default=16)
  ap.add_argument("--time_updates",                   type=int,   default=3)
  ap.add_argument("--time_update_idx",                type=str,   default="k_t")
  ap.add_argument('--prop_separate_params',           action=argparse.BooleanOptionalAction)
  ap.add_argument("--transform_dim" ,                 type=int,   default=10)
  ap.add_argument("--bottle_neck_neurons",            type=int,   default=16)
  ap.add_argument("--bins",                           type=int,   default=16)
  ap.add_argument("--histogram",                      type=bool,  default=False)
  ap.add_argument("--GMN_NPROPLAYERS",                type=int,   default=5)
  ap.add_argument("--MASKING_FOR_MSG_PASSING_COUNT",  type=int,   default=5)
  ap.add_argument("--MARGIN",                         type=float, default=0.1)
  ap.add_argument("--NOISE_FACTOR",                   type=float, default=1.0)
  ap.add_argument("--NUM_RUNS",                       type=int,   default=2)
  ap.add_argument("--BATCH_SIZE",                     type=int,   default=128)
  ap.add_argument("--LEARNING_RATE",                  type=float, default=0.001)
  ap.add_argument("--WEIGHT_DECAY",                   type=float, default=5*10**-4)
  ap.add_argument("--FEAT_TYPE",                      type=str,   default="Onehot1",help="One/Onehot/Onehot1/Adjrow/Adjrow1/AdjOnehot")
  ap.add_argument("--CONV",                           type=str,   default="GCN",help="GCN/GAT/GIN/SAGE")
  ap.add_argument("--DIR_PATH",                       type=str,   default=".",help="path/to/datasets")
  ap.add_argument("--DATASET_NAME",                   type=str,   default="mutag", help="TODO")
  ap.add_argument("--TASK",                           type=str,   default="OurMatchingSimilarity",help="TODO")
  ap.add_argument("--SEED",                           type=int,   default=0)

  av = ap.parse_args()

  if av.FEAT_TYPE == "Adjrow" or  av.FEAT_TYPE == "Adjrow1" or av.FEAT_TYPE == "AdjOnehot": 
      av.TASK = av.TASK + "_" + av.FEAT_TYPE
  if av.CONV != "GCN": 
      av.TASK = av.TASK + "_" + av.CONV
  av.logpath = av.logpath+"_"+av.TASK+"_"+av.DATASET_NAME+"_"+str(av.SEED)+"_"+str(datetime.now()).replace(" ", "_")

  # Print configure
  config = get_default_config()
  config['seed'] = av.SEED
  config['encoder'] ['node_hidden_sizes'] = [10]
  config['encoder'] ['node_feature_dim'] = 1
  config['encoder'] ['edge_feature_dim'] = 1
  config['aggregator'] ['node_hidden_sizes'] = [10]
  config['aggregator'] ['graph_transform_sizes'] = [10]
  config['aggregator'] ['input_size'] = [10]
  config['graph_matching_net'] ['node_state_dim'] = 10
  #config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_matching_net'] ['edge_hidden_sizes'] = [20]
  config['graph_matching_net'] ['node_hidden_sizes'] = [10]
  config['graph_matching_net'] ['n_prop_layers'] = 5
  config['graph_embedding_net'] ['node_state_dim'] = 10
  #config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_embedding_net'] ['edge_hidden_sizes'] = [20]
  config['graph_embedding_net'] ['node_hidden_sizes'] = [10]
  config['graph_embedding_net'] ['n_prop_layers'] = 5
  config['graph_embedding_net'] ['n_prop_layers'] = 5
  config['temporal_gnn'] = {
    'n_time_updates': av.time_updates,
    'time_update_idx': av.time_update_idx,
    'prop_separate_params': av.prop_separate_params
  }
  config['fringe_isonet'] ['masking_for_msg_passing_count'] = av.MASKING_FOR_MSG_PASSING_COUNT
  
  #logger.info("av gmn_prop_param")
  #logger.info(av.GMN_NPROPLAYERS) 
  #logger.info("config param")
  #logger.info(config['graph_embedding_net'] ['n_prop_layers'] )
  config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
  #logger.info("config param")
  #logger.info(config['graph_embedding_net'] ['n_prop_layers'] )

  config['training']['batch_size']  = av.BATCH_SIZE
  config['training']['margin']  = av.MARGIN
  config['evaluation']['batch_size']  = av.BATCH_SIZE
  config['model_type']  = "embedding"
  config['graphsim'] = {} 
  config['graphsim']['conv_kernel_size'] = [10,4,2]
  config['graphsim']['linear_size'] = [24, 16]
  config['graphsim']['gcn_size'] = [10,10,10]
  config['graphsim']['conv_pool_size'] = [3,3,2]
  config['graphsim']['conv_out_channels'] = [2,4,8]
  config['graphsim']['dropout'] = av.dropout 


  # Set random seeds
  seed = config['seed']
  seed_everything(seed)

  av.dataset = av.DATASET_NAME
  test_node_align_node_loss(av, config)


