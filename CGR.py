import torch
import copy
from clnetworks import *
from generator import *
from models import init_weight


class CGRnetwork(NaiveCLnetwork):
    def __init__(self, args):
        super(CGRnetwork, self).__init__(args)
        self.generator_memories = []
        self.generator = GenerativeNetwork(args.dropout, args.channels_num)
        self.generator.apply(init_weight)
        self.generator.to(self.device)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=args.lr)
        self.gloss = 0.0

    def start_task(self):
        super(CGRnetwork, self).start_task()
        self.generator.apply(init_weight)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr)

    def start_epoch(self):
        super(CGRnetwork, self).start_epoch()
        self.gloss = 0.0

    def reservoir_sampling(self, X, y):
        for idx in range(X.shape[0]):
            self.observed_samples += 1
            if self.label_buffer is None:
                self.label_buffer = torch.unsqueeze(y[idx].clone(), dim=0)
                self.buffer_counter += 1
                continue
            if self.buffer_counter < self.buffer_size:
                label = torch.unsqueeze(y[idx].clone(), dim=0)
                self.label_buffer = torch.cat((self.label_buffer, label), dim=0)
                self.buffer_counter += 1
            elif random.random() <= self.buffer_size / self.observed_samples:
                target = random.randint(0, self.buffer_size - 1)
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
        '''start training generator'''
        self.optimizer.zero_grad()
        self.optimizerG.zero_grad()
        X_fake = self.generator(y)
        y_hat = self.net(X_fake)
        L_G = torch.sum(self.loss(y_hat, y.view(-1))) / X_fake.shape[0]
        L_G.backward()
        nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=20, norm_type=2)
        self.optimizerG.step()
        self.gloss += L_G.item()

    def end_epoch(self, valid_dataset):
        train_acc, train_mf1 = self.confusion_matrix.accuracy(), self.confusion_matrix.macro_f1()
        print(f'epoch: {self.epoch}, train loss: {self.train_loss:.3f}, train accuracy: {train_acc:.3f}, '
              f"macro F1: {train_mf1:.3f}, 1000 lr: {self.optimizer.state_dict()['param_groups'][0]['lr'] * 1000:.3f}, "
              f'generator loss: {self.gloss:.3f}')
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
        super(CGRnetwork, self).end_task()
        self.generator_memories.append(copy.deepcopy(self.generator))


if __name__ == '__main__':
    pass