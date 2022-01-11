import torch
import torch.nn as nn
from .position import PositionalEncoding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. InputEmbedding : project the input to embedding size through a lightweight 3D-CNN
        2. PositionalEncoding : adding positional information using sin/cos functions

        sum of both features are output of BERTEmbedding
    """

    def __init__(self, num_features, dropout=0.1):
        """
        :param num_features: number of input features
        :param dropout: dropout rate
        """
        super().__init__()
        channel_size = (32, 64, 256)
        kernel_size = (5, 3, 5, 3)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1,
                      out_channels=channel_size[0],
                      kernel_size=(kernel_size[0], kernel_size[1], kernel_size[1])),
            nn.ReLU(),
            nn.BatchNorm3d(channel_size[0]),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=channel_size[0],
                      out_channels=channel_size[1],
                      kernel_size=(kernel_size[2], kernel_size[3], kernel_size[3])),
            nn.ReLU(),
            nn.BatchNorm3d(channel_size[1]),
        )

        self.linear = nn.Linear(in_features=channel_size[1]*2,
                                out_features=channel_size[2])

        self.embed_size = channel_size[-1]
        self.position = PositionalEncoding(d_model=self.embed_size, max_len=366)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_sequence, doy_sequence):
        batch_size = input_sequence.size(0)
        seq_length = input_sequence.size(1)
        band_num = input_sequence.size(2)
        patch_size = input_sequence.size(3)
        first_dim = batch_size*seq_length

        obs_embed = input_sequence.view(first_dim, band_num, patch_size, patch_size).unsqueeze(1)
        obs_embed = self.conv1(obs_embed)
        obs_embed = self.conv2(obs_embed)
        obs_embed = self.linear(obs_embed.view(first_dim, -1))   # [batch_size*seq_length, embed_size]
        obs_embed = obs_embed.view(batch_size, seq_length, -1)

        position_embed = self.position(doy_sequence)
        x = obs_embed + position_embed   # [batch_size, seq_length, embed_size]

        return self.dropout(x)
