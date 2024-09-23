'''the wrapper combining the encoder and decoder'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext

from encoder import RNNEncoder, Transformer, GRUEncoder
from backbone import Backbone


class classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, num_class=5, dropout=0.1, attention=False, encoder='rnn', embedding_type='glove', TEXT=None):
        super(classifier, self).__init__()
        self.embedding_type = embedding_type
        self.TEXT = TEXT
        self.encoder_type = encoder
        # depending on the embedding type, the input size of the encoder will be different
        if embedding_type != 'glove':
            self.embedding = nn.Embedding(len(TEXT.vocab), input_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=True)
        if encoder == 'rnn':
            self.encoder = RNNEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        elif encoder == 'transformer':
            self.encoder = Transformer(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nhead=10, dropout=dropout)
        elif encoder == 'gru':
            self.encoder = GRUEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.backbone = Backbone(input_size=hidden_size, num_class=num_class,dropout=dropout, attention=attention)
    def forward(self, x):
        device = x.device

        x = self.embedding(x)
        x = x.to(device)

        x = self.encoder(x)
        x = self.backbone(x)
        return x
    def configure_optimizers(self):
        # Adam is chosen as the optimizer as it is widely used and efficient
        return torch.optim.Adam(self.parameters(), lr=0.001)
    # those three functions are designed for the pytorch lightning but due to the torchtext's version conflicts, I directly use my trainer.py
    def training_step(self, batch, batch_idx):
        x = batch.text
        y = batch.label
        output = self(x)
        loss = F.cross_entropy(output, y)
        return loss
    def validation_step(self, batch, batch_idx):
        x = batch.text
        y = batch.label
        output = self(x)
        loss = F.cross_entropy(output, y)
        return loss
    def test_step(self, batch, batch_idx):
        x = batch.text
        y = batch.label
        output = self(x)
        loss = F.cross_entropy(output, y)
        return loss

if __name__ == '__main__':
    # Test Wrapper
    TEXT = torchtext.data.Field(lower=True,fix_length=200,batch_first=True)
    LABEL = torchtext.data.Field(sequential=False)
    train,valid,test = torchtext.datasets.SST.splits(TEXT,LABEL)
    TEXT.build_vocab(train, vectors='glove.6B.100d')
    LABEL.build_vocab(train)
    train_iter, test_iter = torchtext.data.BucketIterator.splits((train, test), batch_size=32)
    wrapper = classifier(input_size=100, hidden_size=256, num_layers=2, num_class=5, dropout=0.1, attention=False, encoder='bert', embedding_type='train', TEXT=TEXT)
    for batch in train_iter:
        x = batch.text
        output = wrapper(x)
        print(output.shape)
        break
    


