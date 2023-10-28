import torch
import torch.nn.functional as F
from subgraph.utils import cudavar
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
import GraphOTSim.python.layers as gotsim_layers

class GraphSim(torch.nn.Module):
    def __init__(self, av, config, input_dim):
        super(GraphSim, self).__init__()
        self.av = av
        self.config = config
        if self.av.FEAT_TYPE == "Onehot1":
          self.input_dim = max(self.av.MAX_CORPUS_SUBGRAPH_SIZE, self.av.MAX_QUERY_SUBGRAPH_SIZE)
        else:
          self.input_dim = input_dim
        self.build_layers()

    def build_layers(self):

        self.gcn_layers = torch.nn.ModuleList([])
        self.conv_layers = torch.nn.ModuleList([])
        self.pool_layers = torch.nn.ModuleList([])
        self.linear_layers = torch.nn.ModuleList([])
        self.num_conv_layers = len(self.config['graphsim']['conv_kernel_size'])
        self.num_linear_layers = len(self.config['graphsim']['linear_size'])
        self.num_gcn_layers = len(self.config['graphsim']['gcn_size'])

        num_ftrs = self.input_dim
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(
                pyg_nn.GCNConv(num_ftrs, self.config['graphsim']['gcn_size'][i]))
            num_ftrs = self.config['graphsim']['gcn_size'][i]

        in_channels = 1
        for i in range(self.num_conv_layers):
            self.conv_layers.append(gotsim_layers.CNNLayerV1(kernel_size=self.config['graphsim']['conv_kernel_size'][i],
                stride=1, in_channels=in_channels, out_channels=self.config['graphsim']['conv_out_channels'][i],
                num_similarity_matrices=self.num_gcn_layers))
            self.pool_layers.append(gotsim_layers.MaxPoolLayerV1(pool_size=self.config['graphsim']['conv_pool_size'][i],
                stride=self.config['graphsim']['conv_pool_size'][i], num_similarity_matrices=self.num_gcn_layers))
            in_channels = self.config['graphsim']['conv_out_channels'][i]

        for i in range(self.num_linear_layers-1):
            self.linear_layers.append(torch.nn.Linear(self.config['graphsim']['linear_size'][i],
                self.config['graphsim']['linear_size'][i+1]))

        self.scoring_layer = torch.nn.Linear(self.config['graphsim']['linear_size'][-1], 1)

    def GCN_pass(self, data):
        features, edge_index = data.x, data.edge_index
        abstract_feature_matrices = []
        for i in range(self.num_gcn_layers-1):
            features = self.gcn_layers[i](features, edge_index)
            abstract_feature_matrices.append(features)
            features = torch.nn.functional.relu(features)
            features = torch.nn.functional.dropout(features,
                                               p=self.config['graphsim']['dropout'],
                                               training=self.training)


        features = self.gcn_layers[-1](features, edge_index)
        abstract_feature_matrices.append(features)
        return abstract_feature_matrices

    def Conv_pass(self, similarity_matrices_list):
        features = [_.unsqueeze(1) for _ in similarity_matrices_list]
        for i in range(self.num_conv_layers):
            features = self.conv_layers[i](features)
            features = [torch.relu(_)  for _ in features]
            features = self.pool_layers[i](features);

            features = [torch.nn.functional.dropout(_,
                                               p=self.config['graphsim']['dropout'],
                                               training=self.training)  for _ in features]
        return features

    def linear_pass(self, features):
        for i in range(self.num_linear_layers-1):
            features = self.linear_layers[i](features)
            features = torch.nn.functional.relu(features);
            features = torch.nn.functional.dropout(features,p=self.config['graphsim']['dropout'],
                                               training=self.training)
        return features

    def forward(self, batch_data,batch_data_sizes,batch_adj):

        q_graphs,c_graphs = zip(*batch_data)
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))
        query_batch = Batch.from_data_list(q_graphs)
        corpus_batch = Batch.from_data_list(c_graphs)

        query_abstract_features_list = self.GCN_pass(query_batch)
        query_abstract_features_list = [pad_sequence(torch.split(query_abstract_features_list[i], list(a), dim=0), batch_first=True) \
                                        for i in range(self.num_gcn_layers)]


        corpus_abstract_features_list = self.GCN_pass(corpus_batch)
        corpus_abstract_features_list = [pad_sequence(torch.split(corpus_abstract_features_list[i], list(b), dim=0), batch_first=True) \
                                          for i in range(self.num_gcn_layers)]

        similarity_matrices_list = [torch.matmul(query_abstract_features_list[i],\
                                    corpus_abstract_features_list[i].permute(0,2,1))
                                    for i in range(self.num_gcn_layers)]

        features = torch.cat(self.Conv_pass(similarity_matrices_list), dim=1).view(-1,
                              self.config['graphsim']['linear_size'][0])
        features = self.linear_pass(features);


        score_logits = self.scoring_layer(features)
        if self.av.is_sig:
          score = torch.sigmoid(score_logits)
          return score.view(-1)
        else:
          return score_logits.view(-1)
