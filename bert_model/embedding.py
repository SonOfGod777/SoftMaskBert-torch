# encoding=utf-8
import torch
from torch import nn
from LayerNorm import *
from config import *


class TokenEmbedding(nn.Module):
    def __init__(self):
        super(TokenEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = LayerNorm(args.eps, args.hidden_size)
        self.token_embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.type_embedding = nn.Embedding(args.type_vocab_size, args.hidden_size)
        self.position_embedding = nn.Embedding(args.max_position_tokens, args.hidden_size)

    def forward(self, input_ids, type_ids=None):  # (batch, sen)
        if type_ids is None:
            type_ids = torch.zeros_like(input_ids)

        mask_tensor = torch.LongTensor(input_ids.size()).fill_(103).to(args.device)
        mask_embedding = self.token_embedding(mask_tensor)

        sen_len = input_ids.size(1)
        position_ids = torch.arange(sen_len, device=args.device).expand_as(input_ids)
        token_embedding = self.token_embedding(input_ids)

        type_embedding = self.type_embedding(type_ids)
        position_embedding = self.position_embedding(position_ids)

        embedding = token_embedding + type_embedding + position_embedding
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        return embedding, mask_embedding


if __name__ == '__main__':
    input_ids = torch.arange(10).view(2, 5)
    TokenEmbedding()(input_ids)








