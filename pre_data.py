# encoding=utf-8
from token_model.tokenization import Tokenizer
from config import *
import torch
import numpy as np
import jieba
import collections


class BuildData(object):
    def __init__(self):
        self.vocab_size = args.vocab_size
        self.Tokenizer = Tokenizer(args.vocab_path)
        self.mask = self.Tokenizer.token_to_id.get('[MASK]')
        self.cls = self.Tokenizer.token_to_id.get('[CLS]')
        self.sep = self.Tokenizer.token_to_id.get('[SEP]')
        self.sentence_len = args.sentence_len

    def load_data(self, path):
        output = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                mis_text, text, labels = line.split('-***-')
                mis_ids = [self.cls] + self.Tokenizer.convert_tokens_to_ids(self.Tokenizer.tokenize(mis_text))
                labels = [0] + [int(i) for i in labels if str(i).strip()]
                positions = [int(i)-1 for i, label in enumerate(labels) if int(label) == 1]
                # 所有字都计算loss
                text_ids = [self.cls] + self.Tokenizer.convert_tokens_to_ids(self.Tokenizer.tokenize(text))
                # 只计算mask字的loss
                label_lis = [token if label == 1 else -100 for label, token in zip(labels, text_ids)]
                print('label_lis', [self.Tokenizer.id_to_token[w] for w in label_lis if w != -100])

                if len(mis_ids) <= self.sentence_len - 1:
                    mis_ids += [self.sep] + (self.sentence_len - len(mis_ids) - 1) * [0]
                    label_lis += [-100] + (self.sentence_len - len(label_lis) - 1) * [-100]
                    labels += [0] + (self.sentence_len - len(labels) - 1) * [0]
                    text_ids += [self.sep] + (self.sentence_len - len(text_ids) - 1) * [0]
                else:
                    mis_ids = mis_ids[:self.sentence_len - 1] + [self.sep]
                    label_lis = label_lis[:self.sentence_len - 1] + [-100]
                    labels = labels[:self.sentence_len - 1] + [0]
                    text_ids = text_ids[:self.sentence_len - 1] + [self.sep]
                output.append((mis_ids, label_lis, labels, positions, text_ids))

        return output

    def token_process(self, token_id):
        rand = np.random.random()
        if rand <= 0.8:
            return self.mask
        if rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocab_size)

    def id_to_mask(self, text_ids):
        index_mask = {}
        covered_token = list()
        num_to_pre = len(text_ids) * 0.15
        label_lis = [str(word).strip() for words in text_ids for word in words]
        token_position = {token: index for index, token in enumerate(label_lis)}
        np.random.seed(1)
        np.random.shuffle(text_ids)

        for words in text_ids:
            if len(index_mask) >= num_to_pre:
                break
            if words in covered_token:
                continue
            covered_token.append(words)
            for word in words:
                position = token_position[str(word).strip()]
                index_mask[position] = self.token_process(word)

        lm_positions = [position for position in index_mask]
        input_lis = [self.cls] + [int(index_mask[index]) if index in index_mask else int(token) for index, token in enumerate(label_lis)]
        label_lis = [-100] + [int(token) if index in index_mask else -100 for index, token in enumerate(label_lis)]

        if len(input_lis) <= self.sentence_len - 1:
            input_lis += [self.sep] + (self.sentence_len - len(input_lis) - 1) * [0]
            label_lis += [-100] + (self.sentence_len - len(label_lis) - 1) * [-100]
        else:
            input_lis = input_lis[:self.sentence_len - 1] + [self.sep]
            label_lis = label_lis[:self.sentence_len - 1] + [-100]
        return input_lis, label_lis, lm_positions

    def build_data(self):
        train_data = self.load_data(args.train_path)
        dev_data = self.load_data(args.dev_path)
        test_data = self.load_data(args.test_path)
        return train_data, dev_data, test_data


class BatchData(object):
    def __init__(self, data, index=0, batch_size=args.batch_size):
        self.index = index
        self.device = args.device
        self.batch_size = batch_size
        self.data = data
        self.batch_nums = len(self.data) // self.batch_size
        self.residue = False
        if len(self.data) % self.batch_size != 0:
            self.residue = True

    def to_tensor(self, batch):
        x = torch.LongTensor([_[0] for _ in batch]).to(self.device)
        y = torch.LongTensor([_[1] for _ in batch]).to(self.device)
        w = torch.LongTensor([_[2] for _ in batch]).to(self.device).float()
        z = [_[3] for _ in batch]
        m = torch.LongTensor([_[4] for _ in batch]).to(self.device)
        return x, y, w, z, m

    def __next__(self):
        if self.residue and self.index == self.batch_nums:
            batch = self.data[self.index*self.batch_size:len(self.data)]
            self.index += 1
            return self.to_tensor(batch)
        elif self.index >= self.batch_nums:
            self.index = 0
            raise StopIteration
        else:
            batch = self.data[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index += 1
            return self.to_tensor(batch)

    def __iter__(self):
        return self


if __name__ == '__main__':
    # path = args.train_path
    pp = BuildData()
    train, _, _ = pp.build_data()
    train_pp = BatchData(train)
    for k in train_pp:
        print('k', k[0])
        print('v', k[1])
        print('w', k[2])
        print('z', k[3])








