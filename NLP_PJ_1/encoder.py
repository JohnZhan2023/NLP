'''Implement two types of encoders: RNN and Transformer for the emotion classification task'''
import torch 
import torch.nn as nn
import torch.nn.functional as F

# RNN Encoder
class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(RNNEncoder, self).__init__()
        # batch normalization will be applied to the input to stabilize the training
        self.bn = nn.BatchNorm1d(input_size)
        # as we use bidirectional LSTM, the hidden size should be divided by 2
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size//2, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
    def forward(self, x):
        bsz = x.size(0)
        seq_len = x.size(1)
        # batch normalization
        x = x.view(bsz*seq_len, -1) # (bsz, seq_len, input_size) -> (bsz*seq_len, input_size)
        x = self.bn(x)        
        x = x.view(bsz, seq_len, -1) # (bsz*seq_len, input_size) -> (bsz, seq_len, input_size)     
        output, _ = self.rnn(x)
        return output
    
class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUEncoder, self).__init__()
        # Batch normalization will be applied to the input to stabilize the training
        self.bn = nn.BatchNorm1d(input_size)
        # Using a bidirectional GRU, the hidden size should be divided by 2 for each direction
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size//2, num_layers=num_layers,
                          dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, x):
        bsz = x.size(0)
        seq_len = x.size(1)
        # Apply batch normalization
        x = x.view(bsz * seq_len, -1)  # Reshape for batch norm: (batch_size * seq_len, input_size)
        x = self.bn(x)
        x = x.view(bsz, seq_len, -1)  # Reshape back: (batch_size, seq_len, input_size)
        # Pass through the GRU
        output, _ = self.gru(x)
        return output
    
# Transformer Encoder
class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nhead, dropout):
        super(Transformer, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size, 
            dropout=dropout
        )
        # stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_layers
        )
        # feedforward layer is added to the output of the transformer
        self.ffc = nn.Linear(input_size, hidden_size)
    def forward(self, x):
        bsz = x.size(0)
        seq_len = x.size(1)
        x = x.view(bsz*seq_len, -1)
        x = self.bn(x)        
        x = x.view(bsz, seq_len, -1)            
        output = self.transformer_encoder(x)
        output = self.ffc(output)
        return output



if __name__ == '__main__':
    # Test RNN Encoder
    rnn_encoder = RNNEncoder(input_size=300, hidden_size=256, num_layers=2, dropout=0.1)
    x = torch.randn(32, 20, 300)
    output = rnn_encoder(x)
    print(output.shape)
    
    # Test Transformer Encoder
    transformer = Transformer(input_size=300, hidden_size=256, num_layers=2, nhead=4, dropout=0.1)
    x = torch.randn(32, 20, 300)
    output = transformer(x)
    print(output.shape)
