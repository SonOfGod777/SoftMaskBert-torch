import torch
from torch import nn
from embedding import *
from config import *
import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# mask 加到 q*v 结果上 ； en_mask:[batch,1,1,leng]  de_mask:[batch,1,leng,leng]
def attention(query, key, value, mask=None, dropout=None):
    attention_hidden_size = query.size(-1)
    score = torch.matmul(query, key.transpose(-1, -2))
    score = score / torch.sqrt(torch.tensor(float(attention_hidden_size)))
    if mask is not None:
        mask = (1.0 - mask.float()) * 1e9
        score -= mask
    prob = nn.Softmax(dim=-1)(score)
    if dropout is not None:
        prob = dropout(prob)
    return torch.matmul(prob, value), prob


class MultiAttention(nn.Module):
    def __init__(self):
        super(MultiAttention, self).__init__()
        self.hidden_size = args.hidden_size
        self.atten_heads = args.mul_attention_heads
        assert self.hidden_size % self.atten_heads == 0
        self.atten_size = self.hidden_size // self.atten_heads
        # self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.linears = clones(nn.Linear(self.hidden_size, self.hidden_size), 4)

    def forward(self, query, key, values, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batchs = query.size(0)
        query, key, values = [linear(x).view(batchs, -1, self.atten_heads, self.atten_size).transpose(1, 2) for linear, x in zip(self.linears, [query, key, values])]
        x, atten_q = attention(query, key, values, mask)
        x = x.transpose(1, 2).contiguous().view(batchs, -1, self.atten_size * self.atten_heads)
        return self.linears[-1](x)


if __name__ == '__main__':
    input_ids = torch.arange(10).view(2, 5)
    input_embedding = TokenEmbedding()(input_ids)
    pp = MultiAttention()(input_embedding, input_embedding, input_embedding)
    print(pp)


