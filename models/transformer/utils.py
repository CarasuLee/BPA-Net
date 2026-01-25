import torch
from torch import nn
from torch.nn import functional as F


def position_embedding(input, d_model):
    input = input.view(-1, 1)
    div_term = torch.arange(0, d_model, 2, dtype=torch.float32, device=input.device)
    div_term = 10000.0 ** (div_term / d_model)
    
    phase = input / div_term
    sin = torch.sin(phase)
    cos = torch.cos(phase)

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, 0:d_model:2] = sin
    len_odd = out[:, 1:d_model:2].shape[1]
    out[:, 1:d_model:2] = cos[:, :len_odd]
    return out


def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out
