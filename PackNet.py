import torch
from clnetworks import *
import numpy as np


class PackNetCLnetwork(CLnetwork):
    def __init__(self, args):
        super(PackNetCLnetwork, self).__init__(args)
        test = torch.zeros(1, args.window_size, 258, 25).to(self.device)
        L = torch.sum(self.net(test))
        L.backward()
        self.grad_number = 0
        self.grad_positions = []
        for param in self.net.parameters():
            if param.grad is not None:
                grad_length = param.grad.view(-1).shape[0]
                self.grad_positions.append((self.grad_number, self.grad_number + grad_length))
                self.grad_number += grad_length
        self.fixed = torch.ones(self.grad_number, dtype=torch.float32, requires_grad=False, device=self.device)
        self.using = torch.zeros(self.grad_number, dtype=torch.float32, requires_grad=False, device=self.device)
        self.proportion = self.grad_number // args.task_num + 1
        self.fixed_numbers = 0
        self.start_fine_tuning = False
        self.using_list = []

    def mask_gradient(self):
        idx = 0
        for param in self.net.parameters():
            starting, ending = self.grad_positions[idx][0], self.grad_positions[idx][1]
            if self.start_fine_tuning:
                target = self.fixed[starting:ending] * self.using[starting:ending]
                target = target.view(param.grad.shape)
            else:
                target = self.fixed[starting:ending].view(param.grad.shape)
            param.grad *= target
            idx += 1

    def mask_params(self):
        idx = 0
        with torch.no_grad():
            for param in self.net.parameters():
                starting, ending = self.grad_positions[idx][0], self.grad_positions[idx][1]
                target = self.using[starting:ending].view(param.data.shape)
                param.data *= target
                idx += 1

    def get_params(self):
        idx = 0
        parameters = torch.zeros(self.grad_number, dtype=torch.float32, requires_grad=False, device=self.device)
        for param in self.net.parameters():
            starting, ending = self.grad_positions[idx][0], self.grad_positions[idx][1]
            parameters[starting:ending].copy_(param.data.view(-1))
            idx += 1
        return parameters

    def sort_params(self):
        parameters = self.get_params().pow(2) * self.fixed
        parameters_left = self.grad_number - self.fixed_numbers
        print(f'flexible parameters left: {parameters_left}')
        if parameters_left <= self.proportion:
            self.using.fill_(1)
        else:
            params_sorted, idx_lst = torch.sort(parameters, descending=True)
            self.using.index_fill_(0, idx_lst[0:self.proportion], 1)
        print('parameter masks updated.')

    def fix_params(self):
        self.fixed_numbers = int(torch.sum(self.using).item())
        self.fixed.copy_(1 - self.using)
        print('parameters fixed.')

    def start_task(self):
        self.epoch = 0
        self.best_net = copy.deepcopy(self.net)
        self.best_train_loss, self.best_train_acc, self.best_valid_acc = 0.0, 0.0, 0.0
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr * 10)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, max(self.args.num_epochs // 6, 1), 0.6)
        self.start_fine_tuning = False

    def start_epoch(self):
        super(PackNetCLnetwork, self).start_epoch()

    def observe(self, X, y, first_time=False):
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        if self.epoch >= self.args.num_epochs // 2 and not self.start_fine_tuning:
            self.start_fine_tuning = True
            print('start fine-tuning...')
            self.sort_params()
        if self.start_fine_tuning:
            self.mask_params()
        y_hat = self.net(X)
        L_current = torch.sum(self.loss(y_hat, y.view(-1)))
        L = L_current / X.shape[0]
        L.backward()
        self.mask_gradient()
        self.optimizer.step()
        self.train_loss += L.item()
        self.confusion_matrix.count_task_separated(y_hat, y, 0)

    def end_epoch(self, valid_dataset):
        train_acc, train_mf1 = self.confusion_matrix.accuracy(), self.confusion_matrix.macro_f1()
        print(f'epoch: {self.epoch}, train loss: {self.train_loss:.3f}, train accuracy: {train_acc:.3f}, '
              f"macro F1: {train_mf1:.3f}, 1000 lr: {self.optimizer.state_dict()['param_groups'][0]['lr'] * 1000:.3f}")
        print(f'flexible numbers: {int(torch.sum(self.fixed).item())}, ', end='')
        if self.start_fine_tuning:
            print(f'using numbers: {int(torch.sum(self.using).item())}')
        else:
            print(f'using numbers: all')
        if (self.epoch + 1) % self.args.valid_epoch == 0:
            print(f'validating on the datasets...')
            valid_confusion = ConfusionMatrix(1)
            valid_confusion = evaluate_tasks(self.net, [valid_dataset], valid_confusion,
                                             self.device, self.args.valid_batch)
            valid_acc, valid_mf1 = valid_confusion.accuracy(), valid_confusion.macro_f1()
            print(f'valid accuracy: {valid_acc:.3f}, valid macro F1: {valid_mf1:.3f}')
            if self.start_fine_tuning and valid_acc > self.best_valid_acc:
                self.best_train_loss = self.train_loss
                self.best_train_acc = train_acc
                self.best_valid_acc = valid_acc
                self.best_net = './modelsaved/' + str(self.args.replay_mode) + '_task' + str(self.task) + '.pth'
                torch.save(self.net.state_dict(), self.best_net)
        self.epoch += 1
        self.scheduler.step()

    def end_task(self):
        super(PackNetCLnetwork, self).end_task()
        self.using_list.append(self.using.clone())
        self.fix_params()


if __name__ == '__main__':
    clnetwork = PackNetCLnetwork(args)
    clnetwork.start_task()
    clnetwork.sort_params()
    print(f'flexible numbers: {int(torch.sum(clnetwork.fixed).item())}, ', end='')
    print(f'using numbers: {int(torch.sum(clnetwork.using).item())}')
    clnetwork.end_task()
    clnetwork.start_task()
    clnetwork.sort_params()
    print(f'flexible numbers: {int(torch.sum(clnetwork.fixed).item())}, ', end='')
    print(f'using numbers: {int(torch.sum(clnetwork.using).item())}')
    clnetwork.end_task()
    clnetwork.start_task()
    clnetwork.sort_params()
    print(f'flexible numbers: {int(torch.sum(clnetwork.fixed).item())}, ', end='')
    print(f'using numbers: {int(torch.sum(clnetwork.using).item())}')
    clnetwork.end_task()
    clnetwork.start_task()
    clnetwork.sort_params()
    print(f'flexible numbers: {int(torch.sum(clnetwork.fixed).item())}, ', end='')
    print(f'using numbers: {int(torch.sum(clnetwork.using).item())}')
    clnetwork.end_task()
