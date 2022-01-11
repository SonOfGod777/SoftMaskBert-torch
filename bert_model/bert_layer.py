import torch
from torch import nn
from RelPosAttention import MultiAttention
from embedding import *
from config import *


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, attention_x):
        dense_x = self.dense1(attention_x)
        dense_x = torch.relu(dense_x)
        dense_x = self.dense2(dense_x)
        return self.dropout(dense_x)


class BertLayer(nn.Module):
    def __init__(self, hidden_size=args.hidden_size, dropout=args.dropout, eps=args.eps, intermediate_size=args.intermediate_size):
        super(BertLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = LayerNorm(eps, hidden_size)
        self.layer_norm2 = LayerNorm(eps, hidden_size)
        self.attention = MultiAttention()
        self.dense = FeedForward(hidden_size, intermediate_size, dropout)

    def forward(self, token_embedding, attention_mask=None):
        attention_x = self.attention(token_embedding, token_embedding, token_embedding, attention_mask)
        attention_x += token_embedding
        attention_x = self.layer_norm1(attention_x)

        dense_x = self.dense(attention_x)
        dense_x += attention_x
        return self.layer_norm2(dense_x)


if __name__ == '__main__':
    input_ids = torch.arange(10).view(2, 5)
    input_embedding = TokenEmbedding()(input_ids)
    pp = BertLayer()(input_embedding)
    print(pp)