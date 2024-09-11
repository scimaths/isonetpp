import torch
from torch import nn

class NeuralTensorNetwork(nn.Module):
    def __init__(self, embedding_dim, score_dim, layers_after_scoring=[]):
        super(NeuralTensorNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.score_dim = score_dim

        self.ntn_bilinear_layer = nn.Bilinear(
            in1_features = embedding_dim,
            in2_features = embedding_dim,
            out_features = score_dim
        )
        self.ntn_concat_layer = nn.Linear(2 * embedding_dim, score_dim)
        self.ntn_activation = nn.ReLU()

        self.post_scoring_layers = nn.Sequential()
        post_scoring_layer_sizes = [score_dim,] + layers_after_scoring
        for idx in range(1, len(post_scoring_layer_sizes)):
            self.post_scoring_layers.append(nn.Linear(
                post_scoring_layer_sizes[idx - 1], post_scoring_layer_sizes[idx]
            ))
            self.post_scoring_layers.append(nn.ReLU())
        self.post_scoring_layers.append(nn.Linear(post_scoring_layer_sizes[-1], 1))

    def forward(self, embedding_1, embedding_2):
        ntn_bilinear_output = self.ntn_bilinear_layer(embedding_1, embedding_2)
        ntn_concat_output = self.ntn_concat_layer(torch.concat([embedding_1, embedding_2], dim=-1))
        activation_output = self.ntn_activation(ntn_bilinear_output + ntn_concat_output)
        return self.post_scoring_layers(activation_output).squeeze(-1)