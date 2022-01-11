# encoding=utf-8
import torch
from torch import nn
import argparse

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

new_key_lis = ['token_embedding.layer_norm.weight', 'token_embedding.layer_norm.bias', 'token_embedding.token_embedding.weight', 'token_embedding.type_embedding.weight', 'token_embedding.position_embedding.weight', 'classify.dense.weight', 'classify.dense.bias', 'bert_layers.##.layer_norm1.weight', 'bert_layers.##.layer_norm1.bias', 'bert_layers.##.layer_norm2.weight', 'bert_layers.##.layer_norm2.bias', 'bert_layers.##.attention.linears.0.weight', 'bert_layers.##.attention.linears.0.bias', 'bert_layers.##.attention.linears.1.weight', 'bert_layers.##.attention.linears.1.bias', 'bert_layers.##.attention.linears.2.weight', 'bert_layers.##.attention.linears.2.bias', 'bert_layers.##.attention.linears.3.weight', 'bert_layers.##.attention.linears.3.bias', 'bert_layers.##.dense.dense1.weight', 'bert_layers.##.dense.dense1.bias', 'bert_layers.##.dense.dense2.weight', 'bert_layers.##.dense.dense2.bias']
old_key_lis = ['bert.embeddings.LayerNorm.gamma', 'bert.embeddings.LayerNorm.beta', 'bert.embeddings.word_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight', 'bert.embeddings.position_embeddings.weight', 'bert.pooler.dense.weight', 'bert.pooler.dense.bias', 'bert.encoder.layer.##.attention.output.LayerNorm.gamma', 'bert.encoder.layer.##.attention.output.LayerNorm.beta', 'bert.encoder.layer.##.output.LayerNorm.gamma', 'bert.encoder.layer.##.output.LayerNorm.beta', 'bert.encoder.layer.##.attention.self.query.weight', 'bert.encoder.layer.##.attention.self.query.bias', 'bert.encoder.layer.##.attention.self.key.weight', 'bert.encoder.layer.##.attention.self.key.bias', 'bert.encoder.layer.##.attention.self.value.weight', 'bert.encoder.layer.##.attention.self.value.bias', 'bert.encoder.layer.##.attention.output.dense.weight', 'bert.encoder.layer.##.attention.output.dense.bias', 'bert.encoder.layer.##.intermediate.dense.weight', 'bert.encoder.layer.##.intermediate.dense.bias', 'bert.encoder.layer.##.output.dense.weight', 'bert.encoder.layer.##.output.dense.bias']

parser = argparse.ArgumentParser(description='transformer xl')
parser.add_argument('--vocab_size', default=21128)
parser.add_argument('--hidden_size', default=768)
parser.add_argument('--num_layers', default=12, help='encoder layers')
parser.add_argument('--gru_num_layers', default=2)
parser.add_argument('--gru_dropout', default=0.8)
parser.add_argument('--mul_attention_heads', default=8)
parser.add_argument('--intermediate_size', default=3072)
parser.add_argument('--max_position_ids', default=1000, help='max sentence long')
parser.add_argument('--type_vocab_size', default=2)
parser.add_argument('--dropout', default=0.9)
parser.add_argument('--eps', default=1e-12, help='layer norm eps')
parser.add_argument('--batch_size', default=10)
parser.add_argument('--epochs', default=100)
parser.add_argument('--learn_rate', default=1e-5)
parser.add_argument('--sentence_len', default=128)
parser.add_argument('--mem_len', default=128)
parser.add_argument('--max_position_tokens', default=512)
parser.add_argument('--device', default=device)
parser.add_argument('--pad_idx', default=0)
parser.add_argument('--classify', default=10)
parser.add_argument('--vocab_path', default='pretrain/vocab.txt')
parser.add_argument('--train_path', default='data/train')
parser.add_argument('--dev_path', default='data/dev')
parser.add_argument('--test_path', default='data/test')
parser.add_argument('--save_path', default='finetune')
parser.add_argument('--pre_path', default='pretrain/pytorch_model.bin')
parser.add_argument('--new_key', default=new_key_lis)
parser.add_argument('--old_key', default=old_key_lis)
parser.add_argument('--num_predict', default=0.15)
parser.add_argument('--gama', default=0.5)
args = parser.parse_args()














