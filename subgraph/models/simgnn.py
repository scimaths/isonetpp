import torch
import torch_geometric.nn as pyg_nn
from subgraph.utils import cudavar
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence 

class SimGNN(torch.nn.Module):
    def __init__(self, av,input_dim):
        """
        """
        super(SimGNN, self).__init__()
        self.av = av
        self.input_dim = input_dim

        #Conv layers
        self.conv1 = pyg_nn.GCNConv(self.input_dim, self.av.filters_1)
        self.conv2 = pyg_nn.GCNConv(self.av.filters_1, self.av.filters_2)
        self.conv3 = pyg_nn.GCNConv(self.av.filters_2, self.av.filters_3)
        
        #Attention
        self.attention_layer = torch.nn.Linear(self.av.filters_3,self.av.filters_3, bias=False)
        torch.nn.init.xavier_uniform_(self.attention_layer.weight)
        #NTN
        self.ntn_a = torch.nn.Bilinear(self.av.filters_3,self.av.filters_3,self.av.tensor_neurons,bias=False)
        torch.nn.init.xavier_uniform_(self.ntn_a.weight)
        self.ntn_b = torch.nn.Linear(2*self.av.filters_3,self.av.tensor_neurons,bias=False)
        torch.nn.init.xavier_uniform_(self.ntn_b.weight)
        self.ntn_bias = torch.nn.Parameter(torch.Tensor(self.av.tensor_neurons,1))
        torch.nn.init.xavier_uniform_(self.ntn_bias)
        #Final FC
        feature_count = (self.av.tensor_neurons+self.av.bins) if self.av.histogram else self.av.tensor_neurons
        self.fc1 = torch.nn.Linear(feature_count, self.av.bottle_neck_neurons)
        self.fc2 = torch.nn.Linear(self.av.bottle_neck_neurons, 1)

    def GNN (self, data):
        """
        """
        features = self.conv1(data.x,data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.av.dropout, training=self.training)

        features = self.conv2(features,data.edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.av.dropout, training=self.training)

        features = self.conv3(features,data.edge_index)
        return features

    def forward(self, batch_data,batch_data_sizes,batch_adj):
        """
          batch_adj is unused
        """
        q_graphs,c_graphs = zip(*batch_data)
        a,b = zip(*batch_data_sizes)
        qgraph_sizes = cudavar(self.av,torch.tensor(a))
        cgraph_sizes = cudavar(self.av,torch.tensor(b))
        query_batch = Batch.from_data_list(q_graphs)
        query_batch.x = self.GNN(query_batch)
        query_gnode_embeds = [g.x for g in query_batch.to_data_list()]
        
        corpus_batch = Batch.from_data_list(c_graphs)
        corpus_batch.x = self.GNN(corpus_batch)
        corpus_gnode_embeds = [g.x for g in corpus_batch.to_data_list()]

        preds = []
        q = pad_sequence(query_gnode_embeds,batch_first=True)
        context = torch.tanh(torch.div(torch.sum(self.attention_layer(q),dim=1).T,qgraph_sizes).T)
        sigmoid_scores = torch.sigmoid(q@context.unsqueeze(2))
        e1 = (q.permute(0,2,1)@sigmoid_scores).squeeze()

        c = pad_sequence(corpus_gnode_embeds,batch_first=True)
        context = torch.tanh(torch.div(torch.sum(self.attention_layer(c),dim=1).T,cgraph_sizes).T)
        sigmoid_scores = torch.sigmoid(c@context.unsqueeze(2))
        e2 = (c.permute(0,2,1)@sigmoid_scores).squeeze()
        
        scores = torch.nn.functional.relu(self.ntn_a(e1,e2) +self.ntn_b(torch.cat((e1,e2),dim=-1))+self.ntn_bias.squeeze())

        #TODO: Figure out how to tensorize this
        if self.av.histogram == True:
          h = torch.histc(q@c.permute(0,2,1),bins=self.av.bins)
          h = h/torch.sum(h)

          scores = torch.cat((scores, h),dim=1)

        scores = torch.nn.functional.relu(self.fc1(scores))
        score = torch.sigmoid(self.fc2(scores))
        preds.append(score)
        p = torch.stack(preds).squeeze()
        return p
