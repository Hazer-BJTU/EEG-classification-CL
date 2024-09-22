from models import *
from metric import *
import copy
import torch


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
        self.train_loss += L
        self.confusion_matrix.count_task_separated(y_hat, y, 0)

    def end_epoch(self, valid_dataset):
        train_acc = self.confusion_matrix.accuracy()
        print(f'epoch: {self.epoch}, train loss: {self.train_loss:.3f}, train accuracy: {train_acc:.3f}')
        if (self.epoch + 1) % self.args.valid_epoch == 0:
            print(f'validating on the datasets...')
            valid_confusion = ConfusionMatrix(1)
            valid_confusion = evaluate_tasks(self.net, [valid_dataset], valid_confusion,
                                             self.device, self.args.valid_batch)
            valid_acc = valid_confusion.accuracy()
            print(f'valid acc: {valid_acc:.3f}')
            if valid_acc > self.best_valid_acc:
                self.best_train_loss = self.train_loss
                self.best_train_acc = train_acc
                self.best_valid_acc = valid_acc
                self.best_net = copy.deepcopy(self.net)
        self.epoch += 1

    def end_task(self):
        self.task += 1
        self.best_net_memory.append(self.best_net)


if __name__ == '__main__':
    pass
