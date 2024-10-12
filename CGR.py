import torch
import math
import copy
from clnetworks import *
from generator import *
from models import init_weight
from torchvision import transforms
unloader = transforms.ToPILImage()


class CGRnetwork(NaiveCLnetwork):
    def __init__(self, args):
        super(CGRnetwork, self).__init__(args)
        self.running_mean, self.running_mean_sqr = 0, 0
        self.running_memory = []

    def start_task(self):
        super(CGRnetwork, self).start_task()
        self.running_mean, self.running_mean_sqr = 0, 0

    def start_epoch(self):
        super(CGRnetwork, self).start_epoch()

    def reservoir_sampling(self, X, y):
        batch_size, seq_length, F, T = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        alpha = self.observed_samples / (self.observed_samples + batch_size)
        beta = batch_size / (self.observed_samples + batch_size)
        '''calculate avg(X) and avg(X^2)'''
        mean = torch.mean(X.view(-1))
        mean_sqr = torch.mean(X.pow(2).view(-1))
        '''update running avg(X) and running avg(X^2)'''
        self.running_mean = alpha * self.running_mean + beta * mean.item()
        self.running_mean_sqr = alpha * self.running_mean_sqr + beta * mean_sqr.item()
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
            print(f'sampling {self.buffer_counter} examples, labels only...')
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
              f"macro F1: {train_mf1:.3f}, 1000 lr: {self.optimizer.state_dict()['param_groups'][0]['lr'] * 1000:.3f}")
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
                self.best_net = './modelsaved/' + str(self.args.replay_mode) + '_task' + str(self.task) + '.pth'
                torch.save(self.net.state_dict(), self.best_net)
        self.epoch += 1
        self.scheduler.step()

    def end_task(self):
        self.task += 1
        self.best_net_memory.append(self.best_net)
        self.memory_buffer.append((None, self.label_buffer))
        mean = self.running_mean
        var = (self.running_mean_sqr - self.running_mean ** 2) * (self.observed_samples / (self.observed_samples - 1))
        var = math.sqrt(var)
        self.running_memory.append((mean, var))


if __name__ == '__main__':
    pass
