import torch
import torch.nn.functional as F
import subgraph.neuromatch as nm
from subgraph.utils import cudavar
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence

class NeuroMatch(nm.OrderEmbedder):
    def __init__(self, input_dim, hidden_dim, av):
        super(NeuroMatch, self).__init__(input_dim, hidden_dim, av)

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        q_graphs,c_graphs = zip(*batch_data)
        query_batch = Batch.from_data_list(q_graphs)
        corpus_batch = Batch.from_data_list(c_graphs)

        query_abstract_features = self.emb_model(query_batch)
        corpus_abstract_features = self.emb_model(corpus_batch)

        return self.predict((query_abstract_features, corpus_abstract_features))

