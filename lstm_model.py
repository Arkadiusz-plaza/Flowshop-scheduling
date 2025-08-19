import torch
import torch.nn as nn

class LSTMSequencer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMSequencer, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  
        out = self.fc(out)   
        return out
