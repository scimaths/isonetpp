import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
import torch.nn.functional as F   
from torch_geometric.nn.glob import global_add_pool
from torch_geometric.data import Batch

class MLPLayers(nn.Module):

    def __init__(self, n_in, n_hid, n_out, num_layers = 2 ,use_bn=True, act = 'relu'):
        super(MLPLayers, self).__init__()
        modules = []
        modules.append(nn.Linear(n_in, n_hid))
        out = n_hid
        use_act = True
        for i in range(num_layers-1):  # num_layers = 3  i=0,1
            if i == num_layers-2:
                use_bn = False
                use_act = False
                out = n_out
            modules.append(nn.Linear(n_hid, out))
            if use_bn:
                modules.append(nn.BatchNorm1d(out)) 
            if use_act:
                modules.append(nn.ReLU())
        self.mlp_list = nn.Sequential(*modules)

    def forward(self,x):
        x = self.mlp_list(x)
        return x


class TensorNetworkModule(torch.nn.Module):

    def __init__(self, tensor_neurons, filters):

        super(TensorNetworkModule, self).__init__()
        self.filters = filters
        self.tensor_neurons = tensor_neurons
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):

        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(
                self.filters, self.filters, self.tensor_neurons
            )
        )
        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(self.tensor_neurons, 2 * self.filters)
        )
        self.bias = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 1))

    def init_parameters(self):

        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):

        batch_size = len(embedding_1)
        scoring = torch.matmul(
            embedding_1, self.weight_matrix.view(self.filters, -1)
        )
        scoring = scoring.view(batch_size, self.filters, -1).permute([0, 2, 1])
        scoring = torch.matmul(
            scoring, embedding_2.view(batch_size, self.filters, 1)
        ).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(
            torch.mm(self.weight_matrix_block, torch.t(combined_representation))
        )
        scores = F.relu(scoring + block_scoring + self.bias.view(-1))
        return scores


class ERIC(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        max_node_set_size,
        max_edge_set_size,
        gnn_filters,
        reduction,
        tensor_neurons,
        dropout,
        device
    ):
        super(ERIC, self).__init__()

        self.filters = gnn_filters
        self.dropout = dropout
        self.device = device
        self.tensor_neurons = tensor_neurons
        self.num_filter = len(self.filters)

        self.gnn_list = nn.ModuleList()
        self.mlp_list_inner = nn.ModuleList()  
        self.mlp_list_outer = nn.ModuleList()  
        self.NTN_list = nn.ModuleList()

        self.gnn_list.append(GINConv(torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.filters[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.filters[0], self.filters[0]),
            torch.nn.BatchNorm1d(self.filters[0]),
        ),eps=True))

        for i in range(self.num_filter-1):
            self.gnn_list.append(GINConv(torch.nn.Sequential(
            torch.nn.Linear(self.filters[i],self.filters[i+1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.filters[i+1], self.filters[i+1]),
            torch.nn.BatchNorm1d(self.filters[i+1]),
        ), eps=True))

        for i in range(self.num_filter):
            self.mlp_list_inner.append(MLPLayers(self.filters[i], self.filters[i],None, num_layers=1,use_bn=False))
            self.mlp_list_outer.append(MLPLayers(self.filters[i], self.filters[i],None, num_layers=1,use_bn=False))
            self.act_inner = F.relu
            self.act_outer = F.relu
            self.NTN = TensorNetworkModule(self.tensor_neurons, self.filters[self.num_filter-1])

        self.channel_dim = sum(self.filters)
        self.reduction = reduction
        self.conv_stack = nn.Sequential(
                            nn.Linear(self.channel_dim, self.channel_dim // self.reduction),
                            nn.ReLU(),
                            nn.Dropout(p = dropout),
                            nn.Linear(self.channel_dim // self.reduction, (self.channel_dim // self.reduction) ),
                            nn.Dropout(p = dropout),
                            nn.Tanh(),
                        )             

        self.score_layer = nn.Sequential(
                            nn.Linear((self.channel_dim // self.reduction) , 16),
                            nn.ReLU(),
                            nn.Linear(16 , 1)
                        )
        self.score_sim_layer = nn.Sequential(
                                    nn.Linear(tensor_neurons, tensor_neurons),
                                    nn.ReLU(),
                                    nn.Linear(tensor_neurons, 1)
                                )

        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

    def convolutional_pass_level(self, enc, edge_index, x):
        feat = enc(x, edge_index)
        feat = F.relu(feat)
        feat = F.dropout(feat, p = self.dropout, training=self.training)
        return feat

    def deepsets_outer(self, batch, feat, filter_idx, size = None):
        size = (batch[-1].item() + 1 if size is None else size)
        pool = global_add_pool(feat, batch, size=size)
        return self.act_outer(self.mlp_list_outer[filter_idx](pool))

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        query_sizes, corpus_sizes = zip(*graph_sizes)
        query_sizes = torch.tensor(query_sizes, device=self.device)
        corpus_sizes = torch.tensor(corpus_sizes, device=self.device)

        query_graphs, corpus_graphs = zip(*graphs)
        query_batch = Batch.from_data_list(query_graphs)
        corpus_batch = Batch.from_data_list(corpus_graphs)

        edge_index_1 = query_batch.edge_index
        edge_index_2 = corpus_batch.edge_index
        features_1 = query_batch.x
        features_2 = corpus_batch.x
        batch_1 = query_batch.batch
        batch_2 = corpus_batch.batch

        conv_source_1 = torch.clone(features_1)
        conv_source_2 = torch.clone(features_2)


        for i in range(self.num_filter):
            conv_source_1 = self.convolutional_pass_level(self.gnn_list[i], edge_index_1, conv_source_1)
            conv_source_2 = self.convolutional_pass_level(self.gnn_list[i], edge_index_2, conv_source_2)

            deepsets_inner_1 = self.act_inner(self.mlp_list_inner[i](conv_source_1)) # [1147, 64]
            deepsets_inner_2 = self.act_inner(self.mlp_list_inner[i](conv_source_2))

            deepsets_outer_1 = self.deepsets_outer(batch_1, deepsets_inner_1,i)
            deepsets_outer_2 = self.deepsets_outer(batch_2, deepsets_inner_2,i)

            diff_rep = torch.exp(-torch.pow(deepsets_outer_1 - deepsets_outer_2, 2)) if i == 0 else torch.cat((diff_rep, torch.exp(-torch.pow(deepsets_outer_1 - deepsets_outer_2,2))), dim = 1)  

        score_rep = self.conv_stack(diff_rep).squeeze()  # (128,64)

        sim_rep = self.NTN(deepsets_outer_1, deepsets_outer_2)

        sim_score = torch.sigmoid(self.score_sim_layer(sim_rep).squeeze())

        score = torch.sigmoid(self.score_layer(score_rep)).view(-1)

        comb_score = self.alpha * score + self.beta * sim_score

        return comb_score