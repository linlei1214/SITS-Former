import torch
import torch.nn as nn
from .bert import BERT


class BERTClassification(nn.Module):
    """
    Downstream task: Satellite Time Series Classification
    """

    def __init__(self, bert: BERT, num_classes):
        """
        :param bert: the BERT-Former model
        :param num_classes: number of classes to be classified
        """

        super().__init__()
        self.bert = bert
        self.classification = MulticlassClassification(self.bert.hidden, num_classes)

    def forward(self, x, doy, mask):
        x = self.bert(x, doy, mask)     # [batch_size, seq_length, embed_size]
        return self.classification(x, mask)


class MulticlassClassification(nn.Module):

    def __init__(self, hidden, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden, num_classes)

    def forward(self, x, mask):
        mask = (1 - mask.unsqueeze(-1)) * 1e6
        x = x - mask        # mask invalid timesteps
        x, _ = torch.max(x, dim=1)      # max-pooling
        x = self.linear(x)
        return x
