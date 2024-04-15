import torch
import numpy as np
from lap import lapjv
import torch.nn.functional as F
from subgraph.utils import cudavar
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence

def dense_wasserstein_distance_v3(cost_matrix):
    lowest_cost, col_ind_lapjv, row_ind_lapjv = lapjv(cost_matrix)

    return np.eye(cost_matrix.shape[0])[col_ind_lapjv]

class GOTSim(torch.nn.Module):
    def __init__(self, av,config,input_dim):
        """
        """
        super(GOTSim, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim

        #Conv layers
        self.conv1 = pyg_nn.GCNConv(self.input_dim, self.av.filters_1)
        self.conv2 = pyg_nn.GCNConv(self.av.filters_1, self.av.filters_2)
        self.conv3 = pyg_nn.GCNConv(self.av.filters_2, self.av.filters_3)
        self.num_gcn_layers = 3

        self.n1 = self.av.MAX_QUERY_SUBGRAPH_SIZE
        self.n2 = self.av.MAX_CORPUS_SUBGRAPH_SIZE
        self.insertion_constant_matrix = cudavar(self.av,99999 * (torch.ones(self.n1, self.n1)
                                                - torch.diag(torch.ones(self.n1))))
        self.deletion_constant_matrix = cudavar(self.av,99999 * (torch.ones(self.n2, self.n2)
                                                - torch.diag(torch.ones(self.n2))))


        self.ot_scoring_layer = torch.nn.Linear(self.num_gcn_layers, 1)

        self.insertion_params, self.deletion_params = torch.nn.ParameterList([]), torch.nn.ParameterList([])
        self.insertion_params.append(torch.nn.Parameter(torch.ones(self.av.filters_1)))
        self.insertion_params.append(torch.nn.Parameter(torch.ones(self.av.filters_2)))
        self.insertion_params.append(torch.nn.Parameter(torch.ones(self.av.filters_3)))
        self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.av.filters_1)))
        self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.av.filters_2)))
        self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.av.filters_3)))

    def GNN (self, data):
        """
        """
        gcn_feature_list = []
        features = self.conv1(data.x,data.edge_index)
        gcn_feature_list.append(features)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.av.dropout, training=self.training)

        features = self.conv2(features,data.edge_index)
        gcn_feature_list.append(features)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.av.dropout, training=self.training)

        features = self.conv3(features,data.edge_index)
        gcn_feature_list.append(features)
        return gcn_feature_list


    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
          batch_adj is unused
        """
        batch_sz = len(batch_data)
        q_graphs,c_graphs = zip(*batch_data)
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))
        query_batch = Batch.from_data_list(q_graphs)
        corpus_batch = Batch.from_data_list(c_graphs)
        query_gcn_feature_list = self.GNN(query_batch)
        corpus_gcn_feature_list = self.GNN(corpus_batch)

        pad_main_similarity_matrices_list=[]
        pad_deletion_similarity_matrices_list = []
        pad_insertion_similarity_matrices_list = []
        pad_dummy_similarity_matrices_list = []
        for i in range(self.num_gcn_layers):

            q = pad_sequence(torch.split(query_gcn_feature_list[i], list(a), dim=0), batch_first=True)
            c = pad_sequence(torch.split(corpus_gcn_feature_list[i],list(b), dim=0), batch_first=True)
            q = F.pad(q,pad=(0,0,0,self.n1-q.shape[1],0,0))
            c = F.pad(c,pad=(0,0,0,self.n2-c.shape[1],0,0))
            #NOTE THE -VE HERE. BECAUSE THIS IS ACTUALLY COST MAT
            pad_main_similarity_matrices_list.append(-torch.matmul(q,c.permute(0,2,1)))

            pad_deletion_similarity_matrices_list.append(torch.diag_embed(-torch.matmul(q, self.deletion_params[i]))+\
                                                    self.insertion_constant_matrix)

            pad_insertion_similarity_matrices_list.append(torch.diag_embed(-torch.matmul(c, self.insertion_params[i]))+\
                                                     self.deletion_constant_matrix)

            pad_dummy_similarity_matrices_list.append(cudavar(self.av,torch.zeros(batch_sz,self.n2, self.n1, \
                                                      dtype=q.dtype)))


        sim_mat_all = []
        for j in range(batch_sz):
            for i in range(self.num_gcn_layers):
                a = pad_main_similarity_matrices_list[i][j]
                b =pad_deletion_similarity_matrices_list[i][j]
                c = pad_insertion_similarity_matrices_list[i][j]
                d = pad_dummy_similarity_matrices_list[i][j]
                s1 = qgraph_sizes[j]
                s2 = cgraph_sizes[j]
                sim_mat_all.append(torch.cat((torch.cat((a[:s1,:s2], b[:s1,:s1]), dim=1),\
                               torch.cat((c[:s2,:s2], d[:s2,:s1]), dim=1)), dim=0))


        sim_mat_all_cpu = [x.detach().cpu().numpy() for x in sim_mat_all]
        plans = [dense_wasserstein_distance_v3(x) for x in sim_mat_all_cpu ]
        mcost = [torch.sum(torch.mul(x,cudavar(self.av,torch.Tensor(y)))) for (x,y) in zip(sim_mat_all,plans)]
        sz_sum = qgraph_sizes.repeat_interleave(3)+cgraph_sizes.repeat_interleave(3)
        mcost_norm = 2*torch.div(torch.stack(mcost),sz_sum)
        scores_new =  self.ot_scoring_layer(mcost_norm.view(-1,3)).squeeze()
        #return scores_new.view(-1)

        if self.av.is_sig:
            return torch.sigmoid(scores_new).view(-1)
        else:
            return scores_new.view(-1)