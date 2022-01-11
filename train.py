# encoding=utf-8
import torch
import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from torch import nn
from torch.optim import Adam
from pre_data import BuildData, BatchData
from config import args
from bert_model.bert_model import BertModel
pd.set_option('display.max_columns', None)
from token_model.tokenization import Tokenizer
Tokenizer = Tokenizer(args.vocab_path)


class Train(object):
    def __init__(self):
        self.train_mask_loss = True
        train_data, dev_data, test_data = BuildData().build_data()
        self.train_data = BatchData(train_data)
        self.dev_data = BatchData(dev_data)
        self.test_data = BatchData(test_data)
        self.BertModel = BertModel()
        self.BertModel.load_pre_model(args.pre_path)
        self.optimizer = Adam(self.BertModel.parameters(), lr=args.learn_rate)
        self.corr_loss = nn.CrossEntropyLoss().to(args.device)
        self.det_loss = nn.BCELoss().to(args.device)


    def train(self):
        total_batch = 0
        last_improve = 0  # 上次loss下降的batch数
        flag = False  # 如果loss很久没有下降，结束训练
        best_loss = float('inf')
        self.BertModel.train()
        for epoch in range(args.epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, args.epochs))
            if flag:
                break
            for i, (trains, label_lis, labels, positions, text_ids) in enumerate(self.train_data):
                if self.train_mask_loss:
                    outputs, sig_res, _ = self.BertModel(trains)
                    outputs_ = outputs.view(args.sentence_len * outputs.size()[0], -1)
                    corr_loss = self.corr_loss(outputs_, label_lis.view(-1))
                    sig_res_ = sig_res.squeeze(-1)
                    det_loss = self.det_loss(sig_res_, labels)
                    loss = args.gama * det_loss + (1-args.gama) * corr_loss

                    self.BertModel.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if total_batch % 1 == 0:
                        outputs = torch.nn.Softmax(dim=-1)(outputs[:, 1:])
                        predict_label = torch.max(outputs.data, -1)[1].cpu()   # 0是最大值，1是最大值索引
                        predict_labels, true_labels = self.gather_index(label_lis, predict_label, positions)
                        train_acc = metrics.accuracy_score(true_labels, predict_labels)
                        dev_acc, dev_loss = self.evaluate(self.BertModel, self.dev_data)
                        if dev_loss < best_loss:
                            best_loss = dev_loss
                            save_path = args.save_path + '/trans_point.ep{}'.format(total_batch)
                            torch.save(self.BertModel.state_dict(), save_path)
                            last_improve = total_batch
                        print('epoch:{}, train_loss:{}, train_acc:{}, dev_loss:{}, dev_acc:{}, last_improve:{}'.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, total_batch))
                    total_batch += 1
                    if total_batch - last_improve > 1000:
                        flag = True
                        break
                else:
                    outputs, sig_res, _ = self.BertModel(trains)
                    outputs_ = outputs.view(args.sentence_len * outputs.size()[0], -1)
                    corr_loss = self.corr_loss(outputs_, text_ids.view(-1))
                    sig_res_ = sig_res.squeeze(-1)
                    det_loss = self.det_loss(sig_res_, labels)
                    loss = args.gama * det_loss + (1 - args.gama) * corr_loss

                    self.BertModel.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if total_batch % 1 == 0:
                        outputs = torch.nn.Softmax(dim=-1)(outputs)
                        predict_label = torch.max(outputs.data, -1)[1].cpu()  # 0是最大值，1是最大值索引
                        # predict_labels, true_labels = self.gather_index(label_lis, predict_label, positions)
                        train_acc = metrics.accuracy_score(true_labels, predict_labels)
                        dev_acc, dev_loss = self.evaluate(self.BertModel, self.dev_data)
                        if dev_loss < best_loss:
                            best_loss = dev_loss
                            save_path = args.save_path + '/trans_point.ep{}'.format(total_batch)
                            torch.save(self.BertModel.state_dict(), save_path)
                            last_improve = total_batch
                        print('epoch:{}, train_loss:{}, train_acc:{}, dev_loss:{}, dev_acc:{}, last_improve:{}'.format(
                            total_batch, loss.item(), train_acc, dev_loss, dev_acc, total_batch))
                    total_batch += 1
                    if total_batch - last_improve > 1000:
                        flag = True
                        break

    def gather_index(self, labels, predict_label, positions):
        true_label = labels[:, 1:].data.cpu()
        true_labels, predict_labels = [], []
        for batch_id in range(predict_label.size()[0]):
            position = torch.tensor(positions[batch_id])
            labels = torch.gather(true_label[batch_id], -1, position).numpy()
            predicts = torch.gather(predict_label[batch_id], -1, position).numpy()
            assert len(labels) == len(predicts)
            true_labels.extend(labels)
            predict_labels.extend(predicts)
        return predict_labels, true_labels

    def gather_len(self, text_ids, predict_label):

        true_label = labels[:, 1:].data.cpu()
        true_labels, predict_labels = [], []
        for batch_id in range(predict_label.size()[0]):
            position = torch.tensor(positions[batch_id])
            labels = torch.gather(true_label[batch_id], -1, position).numpy()
            predicts = torch.gather(predict_label[batch_id], -1, position).numpy()
            assert len(labels) == len(predicts)
            true_labels.extend(labels)
            predict_labels.extend(predicts)
        return predict_labels, true_labels

    def test(self, model, data):
        model.load_state_dict(torch.load(args.save_path))
        model.eval()
        test_acc, test_loss = self.evaluate(model, self.test_data)

    def evaluate(self, model, data):
        model.eval()
        loss_total = 0
        with torch.no_grad():
            true_labels_lis, predict_labels_lis = [], []
            for i, (texts, label_lis, labels, positions, text_ids) in enumerate(data):
                output, sig_res, _ = model(texts)
                # batch_num = output.size()[0]
                # output_ = output.view(args.sentence_len * batch_num, -1)
                # loss = nn.CrossEntropyLoss()(output_, label_lis.view(-1)).to(args.device)
                # loss_total += loss
                output = torch.nn.Softmax(dim=-1)(output[:, 1:])
                predict_label = torch.max(output.data, -1)[1].cpu()  # 0是最大值，1是最大值索引
                predict_labels, true_labels = self.gather_index(label_lis, predict_label, positions)
                print('predict_labels', [Tokenizer.id_to_token[w] for w in predict_labels ])
                print('true_labels', [Tokenizer.id_to_token[w] for w in true_labels ])
                true_labels_lis.extend(true_labels)
                predict_labels_lis.extend(predict_labels)
            train_acc = metrics.accuracy_score(true_labels, predict_labels)
        return train_acc, loss_total/(i+1)


if __name__ == '__main__':
    Train().train()