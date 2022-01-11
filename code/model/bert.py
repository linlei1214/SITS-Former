import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm

from .embedding import BERTEmbedding


class BERT(nn.Module):

    def __init__(self, num_features, hidden, n_layers, attn_heads, dropout=0.1):
        """
        :param num_features: number of input features
        :param hidden: hidden size of the SITS-Former model
        :param n_layers: numbers of Transformer blocks (layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(num_features)

        encoder_layer = TransformerEncoderLayer(hidden, attn_heads, feed_forward_hidden, dropout)
        encoder_norm = LayerNorm(hidden)

        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)

    def forward(self, x, doy, mask):
        mask = mask == 0

        x = self.embedding(input_sequence=x, doy_sequence=doy)

        x = x.transpose(0, 1)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)

        return x
