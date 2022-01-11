import torch.nn as nn
from .bert import BERT


class BERTPrediction(nn.Module):
    """
    Proxy task: missing-data imputation
        Given an incomplete time series with some patches being masked randomly,
        the network is asked to regress the central pixels of these masked patches
        based on the residual ones.
    """

    def __init__(self, bert: BERT, num_features=10):
        """
        :param bert: the BERT-Former model acting as a feature extractor
        :param num_features: number of features of an input pixel to be predicted
        """

        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(self.bert.hidden, num_features)

    def forward(self, x, doy, mask):
        x = self.bert(x, doy, mask)
        return self.linear(x)
