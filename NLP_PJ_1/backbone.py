'''Define the classifier based on the encoder results'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self, input_size, num_class=5,dropout = 0.1, attention=False):
        super(Backbone, self).__init__()
        self.attention = attention
        # pooling layer
        # aggregate the information from the sequence to a single vector
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # the input size of the classifier is (bsz, 200, embedding_size)
        self.fc = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size//2, input_size//4),
            nn.ReLU(),
            nn.Linear(input_size//4, num_class)
        )

        # attention layer
        if self.attention:
            self.attention_layer = nn.MultiheadAttention(embed_dim=input_size, num_heads=8, dropout=dropout)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        if self.attention:
            x = x.permute(1, 0, 2)
            # x: (200, bsz, embedding_size)
            x, _ = self.attention_layer(x, x, x)
            x = x.permute(1, 0, 2)
            # x: (bsz, 200, embedding_size)
        # x: (bsz, 200, embedding_size)
        x = x.permute(0, 2, 1)
        # x: (bsz, embedding_size, 200)
        x = self.pooling(x)
        # x: (bsz, embedding_size, 1)
        x = x.squeeze(-1)
        # x: (bsz, embedding_size)
        x = self.fc(x)
        x = self.softmax(x)

        # x: (bsz, hidden_size)
        return x
