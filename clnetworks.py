from models import *
from metric import *
import copy
import torch
import random


class CLnetwork:
    def __init__(self, args):
        self.args = args
        self.net = SeqSleepNet(args.dropout)
        self.net.apply(init_weight)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.best_train_loss, self.best_train_acc, self.best_valid_acc = 0.0, 0.0, 0.0
        self.train_loss, self.confusion_matrix = 0.0, ConfusionMatrix(1)
        self.best_net = None
        self.best_net_memory = []
        self.device = torch.device(f'cuda:{args.cuda_idx}')
        self.net.to(self.device)
        self.epoch = 0
        self.task = 0

    def start_task(self):
        self.epoch = 0
        self.best_net = copy.deepcopy(self.net)
        self.best_train_loss, self.best_train_acc, self.best_valid_acc = 0.0, 0.0, 0.0

    def start_epoch(self):
        self.train_loss = 0.0
        self.confusion_matrix.clear()
        self.net.train()

    def observe(self, X, y, first_time=False):
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        y_hat = self.net(X)
        L_current = torch.sum(self.loss(y_hat, y.view(-1)))
        L = L_current / X.shape[0]
        L.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
        self.train_loss += L.item()
        self.confusion_matrix.count_task_separated(y_hat, y, 0)

    def end_epoch(self, valid_dataset):
        train_acc, train_mf1 = self.confusion_matrix.accuracy(), self.confusion_matrix.macro_f1()
        print(f'epoch: {self.epoch}, train loss: {self.train_loss:.3f}, train accuracy: {train_acc:.3f}, '
              f'macro F1: {train_mf1:.3f}')
        if (self.epoch + 1) % self.args.valid_epoch == 0:
            print(f'validating on the datasets...')
            valid_confusion = ConfusionMatrix(1)
            valid_confusion = evaluate_tasks(self.net, [valid_dataset], valid_confusion,
                                             self.device, self.args.valid_batch)
            valid_acc, valid_mf1 = valid_confusion.accuracy(), valid_confusion.macro_f1()
            print(f'valid accuracy: {valid_acc:.3f}, valid macro F1: {valid_mf1:.3f}')
            if valid_acc > self.best_valid_acc:
                self.best_train_loss = self.train_loss
                self.best_train_acc = train_acc
                self.best_valid_acc = valid_acc
                self.best_net = copy.deepcopy(self.net)
        self.epoch += 1

    def end_task(self):
        self.task += 1
        self.best_net_memory.append(self.best_net)


class NaiveCLnetwork(CLnetwork):
    def __init__(self, args):
        super(NaiveCLnetwork, self).__init__(args)
        self.buffer_size = args.buffer_size
        self.memory_buffer = []
        self.data_buffer, self.label_buffer = None, None
        self.buffer_counter = 0
        self.observed_samples = 0

    def start_task(self):
        super(NaiveCLnetwork, self).start_task()
        self.data_buffer, self.label_buffer = None, None
        self.buffer_counter = 0
        self.observed_samples = 0

    def start_epoch(self):
        super(NaiveCLnetwork, self).start_epoch()

    def reservoir_sampling(self, X, y):
        for idx in range(X.shape[0]):
            self.observed_samples += 1
            if self.data_buffer is None:
                self.data_buffer = torch.unsqueeze(X[idx].clone(), dim=0)
                self.label_buffer = torch.unsqueeze(y[idx].clone(), dim=0)
                self.buffer_counter += 1
                continue
            if self.buffer_counter < self.buffer_size:
                data = torch.unsqueeze(X[idx].clone(), dim=0)
                label = torch.unsqueeze(y[idx].clone(), dim=0)
                self.data_buffer = torch.cat((self.data_buffer, data), dim=0)
                self.label_buffer = torch.cat((self.label_buffer, label), dim=0)
                self.buffer_counter += 1
            elif random.random() <= self.buffer_size / self.observed_samples:
                target = random.randint(0, self.buffer_size - 1)
                self.data_buffer[target].copy_(X[idx])
                self.label_buffer[target].copy_(y[idx])

    def observe(self, X, y, first_time=False):
        if first_time:
            self.reservoir_sampling(X, y)
            print(f'sampling {self.buffer_counter} examples...')
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        y_hat = self.net(X)
        L_current = torch.sum(self.loss(y_hat, y.view(-1)))
        L = L_current / X.shape[0]
        replay_number = 0
        for sample in self.memory_buffer:
            replay_number += sample[0].shape[0]
        print(f'naively replay on {len(self.memory_buffer)} tasks and {replay_number} examples...')
        for sample in self.memory_buffer:
            X_replay, y_replay = sample[0].to(self.device), sample[1].to(self.device)
            y_hat_replay = self.net(X_replay)
            L_replay = torch.sum(self.loss(y_hat_replay, y_replay.view(-1)))
            L = L + L_replay / replay_number
        L.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
        self.train_loss += L.item()
        self.confusion_matrix.count_task_separated(y_hat, y, 0)

    def end_epoch(self, valid_dataset):
        super(NaiveCLnetwork, self).end_epoch(valid_dataset)

    def end_task(self):
        super(NaiveCLnetwork, self).end_task()
        self.memory_buffer.append((self.data_buffer, self.label_buffer))


if __name__ == '__main__':
    pass
