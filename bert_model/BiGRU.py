import torch
from torch import nn
from config import *
from LayerNorm import LayerNorm


class BiGRU(nn.Module):
    def __init__(self):
        super(BiGRU, self).__init__()
        self.input_size = args.hidden_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.gru_num_layers
        self.dropout = args.gru_dropout
        self.bi_gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=True
        )
        self.bi_gru_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.bi_gru_dense2 = nn.Linear(self.hidden_size, 1)
        self.layer_norm = LayerNorm(args.eps, self.hidden_size)
        self.bi_gru_dropout = nn.Dropout(self.dropout)

    def forward(self, input_ids):
        gru_out, _ = self.bi_gru(input_ids)
        gru_out = self.bi_gru_dense(gru_out)
        gru_out = self.layer_norm(gru_out)
        gru_out = self.bi_gru_dropout(gru_out)
        gru_out = self.bi_gru_dense2(gru_out)
        return gru_out


