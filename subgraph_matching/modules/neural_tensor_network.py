import torch
from torch import nn

class NeuralTensorNetwork(nn.Module):
    def __init__(self, embedding_dim, score_dim, layers_after_scoring=[]):
        super(NeuralTensorNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.score_dim = score_dim

        self.ntn_multi_score_layer = nn.Linear(
            embedding_dim, embedding_dim * score_dim, bias=False
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
        transformed_embedding_1 = self.ntn_multi_score_layer(embedding_1)
        transformed_shape = list(transformed_embedding_1.size()) + [self.embedding_dim]
        transformed_shape[-2] = transformed_shape[-2] // self.embedding_dim
        transformed_embedding_1 = transformed_embedding_1.reshape(*transformed_shape)

        ntn_multi_score_output = torch.matmul(
            transformed_embedding_1,
            embedding_2.unsqueeze(-1)
        ).squeeze(-1)
        ntn_concat_output = self.ntn_concat_layer(torch.concat([embedding_1, embedding_2], dim=-1))
        activation_output = self.ntn_activation(ntn_multi_score_output + ntn_concat_output)
        return self.post_scoring_layers(activation_output).squeeze(-1)