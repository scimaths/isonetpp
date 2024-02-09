import torch
from abc import ABC, abstractmethod

class AlignmentModel(torch.nn.Module, ABC):
    def __init__(self):
        super(AlignmentModel, self).__init__()

    @abstractmethod
    def forward_with_alignment(self, graphs, graph_sizes, graph_adj_matrices):
        pass

    def forward(self, graphs, graph_sizes, graph_adj_matrices):
        return self.forward_with_alignment(graphs, graph_sizes, graph_adj_matrices)[0]